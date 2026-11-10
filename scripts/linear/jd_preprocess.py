#!/usr/bin/env python3
"""Stream JD click events into a reproducible, unscaled sparse utility matrix."""
import argparse
from array import array
import csv
import gzip
import hashlib
import json
from pathlib import Path
import random
import sqlite3
import struct
import sys
import time
from jd_download import REVISION, SOURCE, FILES, sha256

PAPER = [(2000,13541,136877),(5000,15887,265220),(20000,19449,698579),
         (50000,21410,1278391),(200000,23853,2937906),(500000,25214,4679648),
         (1000000,26115,6189526),(2557841,27330,8073837)]
MAGIC = b'JDUTIL01'

def little_bytes(values):
    if sys.byteorder != 'little':
        values.byteswap()
    return values.tobytes()

def aggregate(source, database, source_hash, month):
    if database.exists():
        connection = sqlite3.connect(str(database))
        metadata = dict(connection.execute('SELECT key,value FROM metadata'))
        if metadata['source_sha256'] != source_hash or metadata['month'] != month:
            connection.close()
            raise RuntimeError('Cached click counts use a different source/month; choose a new data directory')
        print('Reusing verified click-count database.', flush=True)
        return connection, metadata
    temporary = database.with_suffix('.building')
    if temporary.exists():
        temporary.unlink()  # Only our incomplete generated cache.
    connection = sqlite3.connect(str(temporary))
    connection.execute('PRAGMA journal_mode=OFF')
    connection.execute('PRAGMA synchronous=OFF')
    connection.execute('PRAGMA cache_size=-262144')
    connection.execute('PRAGMA temp_store=FILE')
    connection.execute('CREATE TABLE clicks (user INTEGER, sku INTEGER, clicks INTEGER NOT NULL, PRIMARY KEY(user,sku)) WITHOUT ROWID')
    sql = 'INSERT INTO clicks VALUES(?,?,?) ON CONFLICT(user,sku) DO UPDATE SET clicks=clicks+excluded.clicks'
    rows = valid = missing = outside = 0
    batch = []
    started = time.monotonic()
    with source.open(newline='',encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        required = {'user_ID','sku_ID','request_time'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'Missing columns: {required-set(reader.fieldnames or [])}')
        for row in reader:
            rows += 1
            user, sku = row['user_ID'].strip(), row['sku_ID'].strip()
            if user in ('','-') or sku in ('','-'):
                missing += 1
                continue
            if month and not row['request_time'].startswith(month):
                outside += 1
                continue
            try:
                uid, sid = int(user,16), int(sku,16)
                if not (0 <= uid < 2**63 and 0 <= sid < 2**63):
                    raise ValueError('identifier exceeds signed 64-bit storage')
            except ValueError as exc:
                raise ValueError(f'Invalid JD identifier on data row {rows}') from exc
            batch.append((uid,sid,1)); valid += 1
            if len(batch) == 100000:
                connection.executemany(sql,batch); batch.clear()
            if valid % 1000000 == 0:
                connection.commit()
                print(f'{rows:,} rows read; {valid:,} identified clicks; {time.monotonic()-started:.1f}s',flush=True)
    if batch:
        connection.executemany(sql,batch)
    connection.execute('CREATE TABLE users AS SELECT user, SUM(clicks) AS total_clicks, COUNT(*) AS distinct_products FROM clicks GROUP BY user')
    connection.execute('CREATE UNIQUE INDEX users_id ON users(user)')
    metadata = dict(source_sha256=source_hash,month=month,input_rows=str(rows),valid_clicks=str(valid),
                    missing_identifier_rows=str(missing),outside_month_rows=str(outside))
    connection.execute('CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)')
    connection.executemany('INSERT INTO metadata VALUES(?,?)',list(metadata.items()))
    connection.commit(); connection.close(); temporary.replace(database)
    return sqlite3.connect(str(database)), metadata

def ranking_report(connection, rank_by):
    # Activity is the number of distinct products clicked; IDs deterministically break ties.
    connection.execute('DROP TABLE IF EXISTS ranked')
    connection.execute(f'CREATE TABLE ranked AS SELECT ROW_NUMBER() OVER (ORDER BY {rank_by} DESC,user ASC) AS rank,user,total_clicks,distinct_products FROM users')
    connection.execute('CREATE UNIQUE INDEX ranked_user ON ranked(user)')
    connection.execute('CREATE UNIQUE INDEX ranked_order ON ranked(rank)')
    connection.commit()

def preprocess(args):
    directory=args.data_dir.resolve(); directory.mkdir(parents=True,exist_ok=True)
    source=(args.source or directory/'JD_click_data.csv').resolve()
    source_hash=sha256(source)
    if not args.fixture and (source.stat().st_size,source_hash) != FILES['JD_click_data.csv']:
        raise ValueError('Source does not match the pinned JD click log; use --fixture only for synthetic tests')
    output=directory/'utilities.bin'; manifest_path=directory/'manifest.json'
    if output.exists() and manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        signature=(manifest.get('source_sha256'),manifest.get('ranking'),manifest.get('budget_seed'),manifest.get('month'))
        if signature==(source_hash,args.rank_by,args.seed,args.month) and sha256(output)==manifest.get('matrix_sha256'):
            print(f'Verified existing {output}',flush=True)
            return
    connection,metadata=aggregate(source,directory/'click_counts.sqlite',source_hash,args.month)
    ranking_report(connection,args.rank_by)
    products=[row[0] for row in connection.execute('SELECT DISTINCT sku FROM clicks ORDER BY sku')]
    if not products:
        raise ValueError('No identified clicks remain after filtering')
    product_index={sku:j+1 for j,sku in enumerate(products)}
    buyers=connection.execute('SELECT COUNT(*) FROM ranked').fetchone()[0]
    nnz=connection.execute('SELECT COUNT(*) FROM clicks').fetchone()[0]
    colptr=array('Q',[1])
    for degree, in connection.execute('SELECT distinct_products FROM ranked ORDER BY rank'):
        colptr.append(colptr[-1]+degree)
    assert len(colptr)==buyers+1 and colptr[-1]==nnz+1
    if len(products)>=2**31 or nnz>=2**31:
        raise ValueError('Native GPU CSR requires 32-bit indexable dimensions/nonzeros')
    header=struct.pack('<8s7Q',MAGIC,1,len(products),buyers,nnz,args.seed,0,0)
    rows_offset=64+8*(buyers+1)
    values_offset=rows_offset+4*nnz
    budgets_offset=values_offset+8*nnz
    temporary=output.with_suffix('.bin.part')
    seen=set(); records=[]; cuts=sorted({x[0] for x in PAPER if x[0]<=buyers}|{buyers})
    expected={n:(m,z) for n,m,z in PAPER}
    next_cut=0; edge_count=0; previous_rank=1
    rowbuf=array('I'); valbuf=array('d'); written=0
    def record_cut(k):
        m,z=expected.get(k,('',''))
        records.append(dict(users=k,goods=len(seen),nnz=edge_count,paper_goods=m,paper_nnz=z,
                            matches_paper=(len(seen)==m and edge_count==z) if m!='' else ''))
    with temporary.open('wb') as stream:
        stream.truncate(budgets_offset+8*buyers)
        stream.seek(0);stream.write(header);stream.write(little_bytes(colptr))
        def flush_buffers():
            nonlocal written,rowbuf,valbuf
            stream.seek(rows_offset+4*written);stream.write(little_bytes(rowbuf))
            stream.seek(values_offset+8*written);stream.write(little_bytes(valbuf))
            written+=len(rowbuf);rowbuf=array('I');valbuf=array('d')
        query='SELECT r.rank,c.sku,c.clicks FROM ranked r JOIN clicks c ON c.user=r.user ORDER BY r.rank,c.sku'
        for rank,sku,count in connection.execute(query):
            if rank!=previous_rank:
                if next_cut<len(cuts) and previous_rank==cuts[next_cut]:
                    record_cut(previous_rank);next_cut+=1
                previous_rank=rank
            rowbuf.append(product_index[sku]);valbuf.append(float(count))
            seen.add(sku);edge_count+=1
            if len(rowbuf)==100000:
                flush_buffers()
        if rowbuf:flush_buffers()
        record_cut(buyers)
        assert written==nnz
        generator=random.Random(args.seed)
        stream.seek(budgets_offset)
        for start in range(0,buyers,100000):
            budget=array('d',(max(generator.random(),2.0**-53) for _ in range(min(100000,buyers-start))))
            stream.write(little_bytes(budget))
    temporary.replace(output)
    with (directory/'goods.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(('matrix_row','sku_ID'))
        writer.writerows((j+1,f'{sku:010x}') for j,sku in enumerate(products))
    with gzip.open(directory/'users.csv.gz','wt',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(('matrix_column','user_ID','total_clicks','distinct_products'))
        writer.writerows((r,f'{u:010x}',c,d) for r,u,c,d in connection.execute('SELECT rank,user,total_clicks,distinct_products FROM ranked ORDER BY rank'))
    with (directory/'instances.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    manifest=dict(format='JDUTIL01',version=1,orientation='goods_by_buyers',goods=len(products),users=buyers,nnz=nnz,
        source=SOURCE,source_revision=REVISION,source_sha256=source_hash,month=args.month,ranking=args.rank_by,
        tie_break='user_ID ascending',utility='raw_click_counts',utility_normalization='none',
        budget_seed=args.seed,budget_rng='Python random.Random MT19937 random()',
        budget_storage='positive unscaled Uniform(0,1) draws',budget_normalization='divide selected budgets by their sum in Julia',
        matrix_sha256=sha256(output),input_statistics=metadata,instances=records,
        paper='https://arxiv.org/pdf/2506.06258',fixture=args.fixture)
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    connection.close()
    print(json.dumps({'goods':len(products),'users':buyers,'nnz':nnz,'instances':records},indent=2),flush=True)
    print(f'Saved raw click utilities to {output}',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=Path(__file__).resolve().parents[2]/'dataset'/'jd')
    parser.add_argument('--source',type=Path)
    parser.add_argument('--seed',type=int,default=1)
    parser.add_argument('--month',default='',help='Optional YYYY-MM filter; default retains the complete published click log')
    parser.add_argument('--rank-by',choices=('total_clicks','distinct_products'),default='distinct_products')
    parser.add_argument('--fixture',action='store_true',help='Allow a small synthetic test CSV instead of the pinned public source')
    args=parser.parse_args()
    if args.seed<0 or args.seed>=2**64:parser.error('seed must fit UInt64')
    preprocess(args)

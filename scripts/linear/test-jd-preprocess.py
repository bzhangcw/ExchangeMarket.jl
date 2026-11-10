#!/usr/bin/env python3
import csv, json, struct, subprocess, sys, tempfile, unittest
from pathlib import Path

class PipelineTest(unittest.TestCase):
    def test_counts_ranking_binary_and_missing_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); source=root/'clicks.csv'
            rows=[('01','0a','2018-03-01'),('01','0a','2018-03-01'),
                  ('02','0a','2018-03-02'),('02','0b','2018-03-02'),
                  ('01','-','2018-03-03'),('03','0c','2018-04-01')]
            with source.open('w',newline='') as f:
                w=csv.writer(f);w.writerow(('sku_ID','user_ID','request_time'));w.writerows(rows)
            command=[sys.executable,str(Path(__file__).with_name('jd_preprocess.py')),
                     '--source',str(source),'--data-dir',str(root),'--fixture']
            subprocess.run(command,check=True,stdout=subprocess.DEVNULL)
            data=(root/'utilities.bin').read_bytes()
            self.assertEqual(struct.unpack('<8s7Q',data[:64]),(b'JDUTIL01',1,3,3,4,1,0,0))
            self.assertEqual(struct.unpack_from('<4Q',data,64),(1,3,4,5))
            self.assertEqual(struct.unpack_from('<4I',data,96),(1,2,2,3))
            self.assertEqual(struct.unpack_from('<4d',data,112),(2.,1.,1.,1.))
            budgets=struct.unpack_from('<3d',data,144)
            self.assertTrue(all(0<x<1 for x in budgets))
            self.assertNotAlmostEqual(sum(budgets),1.) # Julia normalizes each selected cohort.
            manifest=json.loads((root/'manifest.json').read_text())
            self.assertEqual(manifest['utility_normalization'],'none')
            self.assertEqual(manifest['input_statistics']['missing_identifier_rows'],'1')
            self.assertEqual(manifest['users'],3) # Keep boundary-date events from the published file.
            subprocess.run(command,check=True,stdout=subprocess.DEVNULL)
            self.assertEqual(data,(root/'utilities.bin').read_bytes())
            bad=root/'bad.csv';bad.write_text('sku_ID,user_ID,request_time\n01,not-a-hex-id,2018-03-01\n')
            failure=subprocess.run([*command[:2],'--source',str(bad),'--data-dir',str(root/'bad'),'--fixture'],capture_output=True)
            self.assertNotEqual(failure.returncode,0)

if __name__=='__main__':unittest.main()

using CSV, DataFrames, SparseArrays, JLD2

csv_file = "dataset/ml-32m/ratings.csv"

selectsize = 100

# load CSV
# userId,movieId,rating,timestamp
# 1,17,4.0,944249077
# 1,25,1.0,944250228
df = CSV.read(csv_file, DataFrame; header=true)
rename!(df, [:user, :item, :rating, :time])
user_map = Dict(id => i for (i, id) in enumerate(unique(df.user)))
item_map = Dict(id => i for (i, id) in enumerate(unique(df.item)))

# Prepare index vectors
gdf = combine(groupby(df, [:user, :item]), :rating => :sum)
rows = [user_map[id] for id in gdf.user]
cols = [item_map[id] for id in gdf.item]
vals = gdf.sum

# Construct sparse matrix
S = sparse(cols, rows, vals)

@save "ml-32m.jld2" S

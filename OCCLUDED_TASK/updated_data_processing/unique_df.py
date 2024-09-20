from pandas import DataFrame, read_csv, concat


csv41= "/Users/francescoaldoventurelli/Desktop/paziente41_new.csv"
csv45= "/Users/francescoaldoventurelli/Desktop/paziente45_new.csv"

data41, data45 = DataFrame(read_csv(csv41)), DataFrame(read_csv(csv45))

total_df = concat([data45, data41])


# To remove the Unnamed column
total_df.drop(total_df.columns[total_df.columns.str.contains(
    'Unnamed', case=False)], axis=1, inplace=True)

total_df = total_df.reset_index()

total_df.drop(total_df.columns[total_df.columns.str.contains(
    'index', case=False)], axis=1, inplace=True)
#print(total_df)
total_df.to_csv("/Users/francescoaldoventurelli/Desktop/total_csv41_45.csv")
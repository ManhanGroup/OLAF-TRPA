import pandas as pd
import os

def read_params(data_path):
  params = {}
  try:
    # Read data from CSV assuming header=None (no header row)
    data = pd.read_csv(data_path)
    
    ufunc=[]
    for index, row in data.iterrows():
      param_name = row['x']
      params[param_name] =  params.get(param_name, 0)+float(row['Value'])  # Assuming numeric values
      ufunc.append('{}*{}'.format(row['x'],row['Value']))
    print("+".join(ufunc))
  except FileNotFoundError:
    print(f"Error: File not found at {data_path}")
  return params


        
       

def main():
    import os
    model_dir=r".\devmodel"
    for f in os.listdir(model_dir):
       if f.endswith('_model.csv'):
          mfile_path=os.path.join(model_dir,f)
          print(mfile_path)
          read_params(mfile_path)

if __name__=="__main__":
    main()
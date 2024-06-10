import numpy as np
import pandas as pd
import yaml
import sys
rng = np.random.default_rng(12345)

## Load config details from YAML
def load_yaml(filename):
    with open(filename, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as error:
            print(error)
    return None

import time
import random

def read_params(data_path):
  params = {}
  try:
    # Read data from CSV assuming header=None (no header row)
    data = pd.read_csv(data_path)
    
    for k, v in zip(data['x'], data['Value']):
      param_name = k
      params[param_name] =  params.get(param_name, 0)+float(v)  # Assuming numeric values
  except FileNotFoundError:
    print(f"Error: File not found at {data_path}")
  return params


def calculate_utility(params, parcel_frame):
  utility = 0
  for col, value in params.items():
    if col in parcel_frame.columns:
      utility += value * parcel_frame[col]
    
  return utility


class model:
    def __init__(self,config):
        # The initialization code should read a YAML config file and use it to define the basic parameters of the model
        # Examples: number of zones, number of product types
        # Probabaly can use this to load initial supply inventory and capacity values by product type
        self.config = load_yaml(config)
        self.zone_df = pd.read_csv(self.config['zonal_data'])
        self.id = self.config['geo_id']
        self.zone_df.set_index(self.id,drop=False,inplace=True)
        self.neighbors=np.load(self.config['neighbors'])
        self.year=self.config['year']
        self.land_uses = self.config['land_uses']
        self.draws = self.config['draws']
        self.qry_developable=self.config['filter_Undevelopable']

        
    def sample_alts(self,LU):
        # This method samples development options for a land use, respecting filters
        sites_avail = self.zone_df.query(self.land_uses[LU]["filter_fn"] + " & "+ self.qry_developable) 
        site_sample = sites_avail.sample(self.draws,replace=False,axis=0)
        return(site_sample)
    def allocate(self):
        start = time.time()
        # This method allocates land use control totals using Monte Carlo simulation
        id = self.config['geo_id']
        #dev_queue = [] # enumerate a "queue" of development projects to build
        to_allocate = 0
        position = 0
        progress = 0
        _last_part = 0
        print("Allocating queue...")

        for LU in self.land_uses:
           to_allocate += int(self.land_uses[LU]["total"])
           store_fld = self.land_uses[LU]["store_fld"]
           self.zone_df[store_fld] = 0 # initialize field to which units will be allocated
           print("Total {} units of {} to allocate".format(self.land_uses[LU]['total'],self.land_uses[LU]['name']))

        while len(self.land_uses)>0:
            LU = random.choice(list(self.land_uses.keys()))
            if self.land_uses[LU]["total"]==0:
               self.land_uses.pop(LU, None)
               continue
            store_fld = self.land_uses[LU]["store_fld"]
            self.zone_df[store_fld] = 0 # initialize field to which units will be allocated
            #print("Enumerating " + self.land_uses[LU]['name'] + " to allocate")
            
            store_fld = self.land_uses[LU]["store_fld"]
            options = self.sample_alts(LU)
            params = read_params(self.land_uses[LU]['value_par'])
            utility = calculate_utility(params, options)
            #utility = options.eval(value_fn,inplace=False).to_numpy()
            expUtil = np.exp(utility)
            denom = np.sum(expUtil)
            probs = expUtil/denom
            zoneSel = rng.choice(options.index,p=probs.fillna(0))
            if self.land_uses[LU]["capacity_fn"]==1:
               alloc = 1
            else: 
               alloc=int(min(self.land_uses[LU]["total"],np.ceil(self.zone_df.loc[self.zone_df[id]==zoneSel].eval(self.land_uses[LU]["capacity_fn"],inplace=False).squeeze())))
            self.zone_df.at[zoneSel,store_fld] += alloc
            self.land_uses[LU]["total"]=self.land_uses[LU]["total"]-alloc

            #remaining = int(self.land_uses[LU]["total"])
            #print("Remaining {} {}  to allocate".format(self.land_uses[LU]['total'] , self.land_uses[LU]['name']))
            if self.land_uses[LU]["total"]==0:
               self.land_uses.pop(LU, None)            
            
            position = position + alloc
            progress = round(100*(position)/to_allocate,0)
            part = round((progress % 10)/2,0)
            if part != _last_part:
                if part == 0:
                    print(f"{progress}%")
                else:
                    print(".", end="", flush=True)

            _last_part = part

        
        run_min = round((time.time()-start)/60,1)
        print(f"Total run time = {run_min} minutes")
    def update(self):
        # this is a function for custom updates on the dataframe before/after allocation
        for op in self.config['update_block']:
            self.zone_df.eval(op, inplace=True)
    
    def updateneighbor(self):
       # Load neighbors from the NPY file
       # remove the address/point itself from the array because it itself is its nearest neighbour
       neighbors = self.neighbors[:, 1:]
       self.zone_df.loc[(self.zone_df.BUILDING_SQFT>0), 'bldage']=self.zone_df.loc[(self.zone_df.BUILDING_SQFT>0),'YEAR_BUILT'].map(lambda x: self.year-x)

       #Create a dataframe to store intermediate columns
       nei_parcel = pd.DataFrame(index=self.zone_df.index)
       ncols=['neighbors_age','neighbors_totalunits','neighbors_BUILDING_SQFT','neighbors_PARCEL_SQFT','neighbors_per_built']
       for c in ncols:
           nei_parcel[c]=None

       #calculate neiboring parcels total building ages
       nei_parcel['neighbors_age']= [self.zone_df['bldage'].iloc[n].sum() for n in neighbors]

       #calculate neiboring parcels total building sqft and total parcel sqft
       nei_parcel['neighbors_BUILDING_SQFT']= [self.zone_df['BUILDING_SQFT'].iloc[n].sum() for n in neighbors]
       nei_parcel['neighbors_PARCEL_SQFT']= [self.zone_df['PARCEL_SQFT'].iloc[n].sum() for n in neighbors]

       #calculate neiboring parcels percent with building_sqft>0
       nei_parcel['neighbors_per_built']= [self.zone_df['BUILDING_SQFT'].iloc[n].map(lambda x: 1 if x>0 else 0).sum()/8.0 for n in neighbors]
       
       #calculate neiboring parcels total units
       nei_parcel['neighbors_totalunits']= [self.zone_df['UNITS'].iloc[n].sum() for n in neighbors]
       
       self.zone_df['neighbors_FAR']=nei_parcel['neighbors_BUILDING_SQFT']/nei_parcel['neighbors_PARCEL_SQFT'].map(lambda x: x if x>0 else 1) 
       self.zone_df['neighbors_bldsqft_per_unit']=nei_parcel['neighbors_BUILDING_SQFT']/nei_parcel['neighbors_totalunits'].map(lambda x: x if x>0 else 1)
       self.zone_df['neighbors_unit_per_acre']=nei_parcel['neighbors_totalunits']*43560.0/nei_parcel['neighbors_PARCEL_SQFT'].map(lambda x: x if x>0 else 1)
       
       self.zone_df['neighbors_bldsqft_per_unit_n']=np.log(self.zone_df['neighbors_bldsqft_per_unit'].map(lambda x: 1 if (x==0) | (x is None) else x))/np.log(self.zone_df['neighbors_bldsqft_per_unit'].max())
       self.zone_df['neighbors_age_n']=self.zone_df['neighbors_age']/(self.zone_df['neighbors_per_built'].map(lambda x: 1 if (x==0) | (x is None) else x)*8)
       self.zone_df['neighbors_age_n']=self.zone_df['neighbors_age_n']/self.zone_df['neighbors_age_n'].max()
   
        
       

def main():
    test_model = model(sys.argv[1])
    test_model.allocate()
    test_model.update()
    test_model.updateneighbor()
    test_model.zone_df.to_csv(sys.argv[2], index=False)

if __name__=="__main__":
    main()
import pandas as pd


dataset=pd.read_csv('data_moods.csv') #import

## The dataset has been taken from kaggle

def music_select(val):
    
    ## randomly select a datarow based on the given emotion
    
    df=dataset[dataset.mood==val].sample(n=1)
    
    print(df['mood'])
    
    ## extract the name of the song and return it to the calling function
    
    res=df.iloc[0]['name']
    
    return str(res)

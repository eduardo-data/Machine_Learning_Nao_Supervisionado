from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=Warning) 
sns.set_style("ticks")
sns.set_context("talk")


class MethodsPreparation():
    
    """Class with some functions for the project. Explanations within each method."""
    
    def __init__(self):
        pass
    
    def preprocessing(self, df):
        
        '''Data pre-processing using StandardScale'''
        
        scaler = StandardScaler()
        scaled_data = pd.DataFrame(scaler.fit_transform(df),
                            columns=df.columns,
                            index=df.index)
        return scaled_data
    
    
    def plot_variables(self, df_scaled):
        
        """Visualization of data before finalizing preparation."""
    
        Numerical = df_scaled
        sns.set_palette("light:g")
        plt.figure(figsize = (15,25))
        for idx, i in enumerate(Numerical):
            plt.subplot(8, 2, idx + 1)
            sns.boxplot(x = i, data = df_scaled)
            plt.title(i,backgroundcolor='black',color='white',fontsize=15)
            plt.xlabel(i, size = 12)
        plt.tight_layout()  
                           
        return plt.show();
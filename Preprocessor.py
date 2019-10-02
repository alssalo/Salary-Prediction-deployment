import pandas as pd
"""
The pipeline object called my_model implements fit and predict methods. When we call the fit method,
the pipeline will execute preprocessor.fit_transform() on the data we pass in the arguments,
then pass that result to PolynomialFeatures.fi_transform()
and finally pass the results to The LinearRegression.fit().
Similarly, when we call the predict method,
it will execute preprocessor.transform() and PolynomialFeatures.transform() and then LinearRegression.predict().

"""
class Preprocessor:
    
    def __init__(self):
        self.was_fit = False
    
    def feature_target_merger(self,X,y):
        return pd.merge(X,y,left_index=True, right_index=True)
    
    def feature_splitter(self,df):
        new_cols_df=df[["cat_feat_mean","exp_dist_mean","dist_job_mean"]]
        df.drop("salary",axis=1,inplace=True)
        return(df,new_cols_df)
        
        
        
    def one_hot_encoding(self,df):
        return pd.get_dummies(df)
        
    def Attended_College(self,df):
        if(df.degree=="NONE" or df.degree=="HIGH_SCHOOL"):
            return "No"
        else:
            return "Yes"
        
    def job_level(self,df):
        if (df.jobType=="JANITOR"): 
            return "JANITOR"

        if((df.jobType=="JUNIOR")or(df.jobType=="SENIOR")):
            return "Average Employees"
    
        if((df.jobType=="MANAGER")or(df.jobType=="VICE_PRESIDENT")):
            return "Management"

        if((df.jobType=="CEO")or(df.jobType=="CTO")or(df.jobType=="CFO")):
            return "Executives"
        
    def dist_binning(self,df):
        return pd.cut(df.milesFromMetropolis,bins=[-1,10,20,30,40,50,60,70,80,90,100],labels=["0","1","2","3","4","5","6","7","8","9"])
        
    def exp_binning(self,df):
        return pd.cut(df.yearsExperience,bins=[-1,6,12,18,25],labels=["0-6","7-12","13-18","19-25"])

        
    def feature_workshop(self,df,feat_list,col_name):
        group=df.groupby(feat_list)
        mean_encode=pd.DataFrame({col_name: group["salary"].mean()})
        df = pd.merge(df, mean_encode, on=feat_list, how='left')
        return df
    
    def feature_engineerer(self,df):
        df=self.feature_workshop(df,['jobType', 'degree', 'major', 'industry'],"cat_feat_mean")
        df=self.feature_workshop(df,["Experience_bracket","dist_bracket"],"exp_dist_mean")
        #df=feature_workshop(df,["Experience_bracket","job_level","Attended_College"],"exp_job_cllg_mean")
        df=self.feature_workshop(df,["dist_bracket","job_level"],"dist_job_mean")
        return df
        
    def drop_columns(self,df):
        df.drop(["dist_bracket","job_level","Experience_bracket"],axis=1,inplace=True)
        return df
    
            
    def fit(self, X, y=None):
        
        self.was_fit=True
        
        X_new=X.copy()
        #feature generation
        X_new["Attended_College"]=X_new.apply(self.Attended_College, axis=1)
        X_new["job_level"]=X_new.apply(self.job_level, axis=1)
        X_new["dist_bracket"]=self.dist_binning(X_new)
        X_new["Experience_bracket"]=self.exp_binning(X_new)
        
        #feature engineereer
        X_new=self.feature_target_merger(X_new,y)
        X_new=self.feature_engineerer(X_new)
        X_new,self.new_col_df=self.feature_splitter(X_new)
        
        
        del X_new
        
        return self
        
    def transform(self,X,y=None):
        
        if not self.was_fit:
            raise Error("need to fit preprocessor first")
        X_new=X.copy()    
        #feature generation
        X_new["Attended_College"]=X_new.apply(self.Attended_College, axis=1)
        X_new["job_level"]=X_new.apply(self.job_level, axis=1)
        X_new["dist_bracket"]=self.dist_binning(X_new)
        X_new["Experience_bracket"]=self.exp_binning(X_new)
        
        #feature engineereer
        
        X_new=pd.merge(X_new,self.new_col_df,left_index=True, right_index=True)
        
        #drop columns
        X_new=self.drop_columns(X_new)
        
        #encoding
        X_new=self.one_hot_encoding(X_new)
  
        
        return X_new#,y_new
    
    def fit_transform(self, X, y=None):
        """fit and transform wrapper method, used for sklearn pipeline"""

        return self.fit(X,y).transform(X,y)

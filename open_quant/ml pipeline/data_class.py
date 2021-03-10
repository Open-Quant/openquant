import pandas as pd
import boto3


class DataFrame:
    # Initialize DataFrame Object
    def __init__(self, df, csv_name):
        self.df = df
        self.csv_name = csv_name
        self.final_name = ""

    # Upload to aws
    def upload_to_aws(self, local_file, bucket="capitalprawn-sagemaker-data", object_name="None"):
        s3 = boto3.client('s3')
        print('uploading {} to aws.'.format(local_file))
        s3.upload_file(local_file, bucket, "bars/" + local_file)
        print('{} uploaded to aws.'.format(local_file))

    # Add headers
    def add_headers(self, header_list):
        self.df.columns = header_list

    # Drop Columns
    def drop_columns(self, column_list):
        self.df.drop(columns=column_list, inplace=True)

    # Drop NA values from a specified column
    def drop_na(self, column_header):
        self.df.dropna(column_header, inplace=True)

    # Fill NA with Median values
    def median_fill(self):
        median = self.df[1].median()
        self.df[1].fillna(median, inplace=True)
        print("filled na values")

    # Write the contents of our current df to a local csv
    # Final path is also saved here. This final path stores the path for our modified data.
    # The final path will be used by the upload_to_aws() method.
    def write_df_to_csv(self):
        self.final_name = "modified-" + self.csv_name
        self.df.to_csv(self.final_name, index=False)

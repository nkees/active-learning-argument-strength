import pandas as pd
import os


class FileManager:
    """
    A class for quick management of file operations.
    """

    def __init__(self, file_path, file_name):
        self.path = file_path
        self.name = file_name

    def read(self):
        return pd.read_csv(os.path.join(self.path, self.name))

    def exists(self):
        if os.path.exists(os.path.join(self.path, self.name)):
            return True
        else:
            return False

    def save_with_replacement(self, df, index=False):
        df.to_csv(
            os.path.join(self.path, self.name),
            index=index
        )

    def append_and_save(self, df):
        if self.exists():
            latest_samples = self.read()
            df_appended = pd.concat([latest_samples, df])
            self.save_with_replacement(df_appended)
            return df_appended
        else:
            self.save_with_replacement(df)
            return df

    def get_ids(self):
        if self.exists():
            data_set = self.read()
            try:
                id_list = data_set["id"].to_list()
            except IndexError:
                raise IndexError('The given file seems to have no column "id". Please check if the file has been specified correctly.')
            return id_list
        else:
            raise FileNotFoundError("The given file does not exist.")

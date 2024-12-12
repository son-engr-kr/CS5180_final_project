from package.mongodb_handler import MongoHandler
from datetime import datetime
from bson import ObjectId
from stable_baselines3.common.base_class import BaseAlgorithm
import os
class TrainSession:
    KEY_LIST = ["model_file_path","num_timesteps"]
    def __init__(self, *, db_name:str, session_config:dict=None, session_id:ObjectId|None=None, find_prev_train_session:bool=True):
        self.mongo_handler = MongoHandler(db_name, "train_sessions")
        if session_id is not None:
            self.session_id:ObjectId = session_id
        elif find_prev_train_session:
            if session_config is not None:
                self.session_id:ObjectId = self.mongo_handler.get_prev_session_id_or_none(session_config)
            else:
                raise ValueError("Either 'session_id' or 'session_info' must be provided")
        else:
            self.session_id = None
        
        # print(f"self.train_session_id:{self.train_session_id} ({type(self.train_session_id)})")
        is_train_session_exist = self.session_id is not None

        if is_train_session_exist:
            print(f"prev train session exists")

            last_train_log = self.mongo_handler.get_last_train_log(self.session_id)
            if last_train_log is not None:
                last_model_path = last_train_log["model_file_path"]
                last_num_timesteps = last_train_log["num_timesteps"]
                print(f"last model path: {last_model_path}")

            else:
                print("train log not exists.")
        else:
            print(f"make new train session")

            self.session_id:ObjectId = self.mongo_handler.make_new_train_session(session_config)
    def get_session_config(self)->dict:
        return self.mongo_handler.get_session_by_id_without_train_log(self.session_id)["session_config"]
        
        
    def make_check_point(self, *,
                         model:BaseAlgorithm,
                         log_dict:dict):
        
        logs_dir = os.path.join('./rl_model_checkpoints', f"{self.session_id}")
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        model_path = os.path.join(logs_dir,f"step_{model.num_timesteps}")
        model.save(model_path)
        print(f"model saved at {model_path}")

        log_dict["model_file_path"] = model_path
        self.mongo_handler.add_train_log(self.session_id, log_dict)
    # def get_last_train_log(self)->tuple[int,str]:
    #     return self.get_specific_index_train_log(-1)
    # def get_specific_index_train_log(self, index:int)->tuple[int,str]:
    #     train_log =  self.mongo_handler.get_specific_index_train_log(self.session_id, index)
    #     if train_log is not None:
    #         return train_log["num_timesteps"], train_log["model_file_path"]
    #     else:
    #         return None,None
    def get_last_train_log(self)->any:
        return self.get_specific_index_train_log(-1)
    def get_specific_index_train_log(self, index:int)->any:
        train_log =  self.mongo_handler.get_specific_index_train_log(self.session_id, index)
        return train_log

        
                
if __name__ == "__main__":
    print("this is train_handler test")
from pymongo import MongoClient
from pprint import pprint
import time
import os
from bson import ObjectId
class MongoHandler:
    def __init__(self, db_name: str, collection_name:str):
        self.client = MongoClient('mongodb://localhost:27017/')
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
    ##### about train session #####
    def get_prev_session_or_none(self, session_config: dict):
        return self.collection.find_one({"session_config": session_config})
    def get_prev_session_id_or_none(self, session_config: dict)->ObjectId:
        last_session = self.collection.find_one({"session_config": session_config}, {"_id": 1})
        if last_session is None:
            return None
        return self.collection.find_one({"session_config": session_config}, {"_id": 1})["_id"]
    def make_new_train_session(self, session_config: dict)->ObjectId:
        document = {"session_config": session_config, "train_log": []}
        session_id = self.collection.insert_one(document).inserted_id
        # return self.train_sessions.find_one({"_id": session_id})
        return session_id

    def add_train_log(self, session_id, train_log: dict):
        self.collection.update_one(
            {"_id": session_id},
            {"$push": {"train_log": train_log}}
        )
    def get_session_by_id(self, session_id:ObjectId):
        session = self.collection.find_one({"_id": session_id})
        if session:
            return session
        return None
    def get_session_by_id_without_train_log(self, session_id:ObjectId):
        session = self.collection.find_one({"_id": session_id}, {"train_log": 0})
        if session:
            return session
        return None
    def get_train_log_list(self, session_id:ObjectId):
        session = self.collection.find_one({"_id": session_id}, {"train_log": 1})
        if session and "train_log" in session and session["train_log"]:
            return session["train_log"]
        return None
    def get_last_train_log(self, session_id:ObjectId):
        return self.get_specific_index_train_log(session_id,-1)
    def get_specific_index_train_log(self, session_id:ObjectId, index:int):
        train_log_list = self.get_train_log_list(session_id)
        if train_log_list is not None:
            return train_log_list[index]
        return None
    def get_latest_session_id(self)->ObjectId:
        latest_session = self.collection.find_one(sort=[("_id", -1)])
        if latest_session:
            return latest_session["_id"]
        return None
    

    def delete_last_n_train_logs(self, session_id:ObjectId, n):
        session = self.collection.find_one({"_id": session_id})
        print(session["train_log"])
        if session and "train_log" in session and session["train_log"]:
            train_log_list = session["train_log"]
            print(len(train_log_list))
            for _ in range(n):
                train_log_list.pop()
            print(len(train_log_list))
            
            self.collection.update_one(
                {"_id": session_id},
                {"$set": {"train_log": train_log_list}}
            )

    def delete_last_n_train_logs2(self, session_id:ObjectId, n):
        session = self.collection.find_one({"_id": session_id})
        print(session["train_log"])
        if session and "train_log" in session and session["train_log"]:
            for _ in range(n):
                self.collection.update_one(
                    {"_id": session_id},
                    {"$pop": {"train_log": 1}}
                )
    
        
    def delete_logs_until_file(self, session_id:ObjectId):
        train_logs = self.get_train_log_list(session_id)
        if train_logs is None:
            return
        for i in range(len(train_logs) - 1, -1, -1):
            if train_logs[i]["model_file_path"] is not None and os.path.exists(train_logs[i]["model_file_path"]):
                break
            self.delete_last_n_train_logs(session_id, 1)

    #####! about train session #####

    #####  about train archive #####
    def make_new_train_archive(self, session_config_combinations: dict)->ObjectId:
        document = {"session_config_combinations": session_config_combinations, "session_logs": []}
        archive_id = self.collection.insert_one(document).inserted_id
        # return self.train_sessions.find_one({"_id": session_id})
        return archive_id
    def add_session_log_to_archive(self, archive_id, session_log: dict):
        self.collection.update_one(
            {"_id": archive_id},
            {"$push": {"session_logs": session_log}}
        )

    def get_archive_by_id(self, archive_id:ObjectId):
        session = self.collection.find_one({"_id": archive_id})
        if session:
            return session
        return None
if __name__ == "__main__":
    mongo_handler = MongoHandler("test", "train_sessions")
    train_version = {"rl_version": "240326-5", "model": "pytorch 모델을 식별가능한 식별자2"}
    last_session = mongo_handler.get_prev_session_or_none(train_version)
    last_session_id = mongo_handler.get_prev_session_id_or_none(train_version)
    print("last_session: ",end="");pprint(last_session)
    print("last_session_id: ",end="");pprint(last_session_id)
    if last_session is None:
        last_session = mongo_handler.make_new_train_session(train_version)
    # session_id = last_session["_id"]
    
    last_train_info = mongo_handler.get_last_train_log(last_session_id)
    if last_train_info:
        start_episode = last_train_info["episode"] + 1
        start_total_step = last_train_info["total_step"] + 1
        last_file_path = last_train_info["file_path"]
    else:
        start_episode = 0
        start_total_step = 0
        last_file_path = None

    for idx in range(start_episode, start_episode + 10):
        train_info = {
            "episode": idx,
            "total_step": start_total_step + idx - start_episode,
            "time": time.time(),
            "reward": 3.9454,
            "loss": 3.83432,
            "file_path": None if (idx+1) % 10 != 0 else f"torch_models/240326_0439_{idx}.pth",
        }
        mongo_handler.add_train_log(last_session_id, train_info)

    import torch
    import torch.nn as nn
    import torch.optim as optim
    class TestModel(nn.Module):
        def __init__(self, state_dim, action_dim):
            super(TestModel, self).__init__()
            self.actor_mu = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, action_dim),
                nn.Tanh(),
            )
            self.actor_std = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, action_dim),
                nn.Sigmoid(),
            )
            
            self.critic = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )
            # self.log_std_linear = nn.Linear(256, action_dim)
            # self.log_std = nn.Parameter(torch.zeros(action_dim))

        def forward(self, state):
            action_mean = self.actor_mu(state)
            action_std = self.actor_std(state)
            state_value = self.critic(state)

            return action_mean, action_std, state_value
    model = TestModel(3,4)
    print(f"model: {str(model)}")


    optimizer = optim.Adam(model.parameters(), lr=0.03)
    print(f"optimizer: {str(optimizer)}")

    ## dict test
    dict1 = {
        "hello1":1,
        "hello2":"2",
    }
    dict2 = {
        "hello1":1,
        "hello2":"2",
    }
    dict3 = {
        "hello1":1,
        "hello2":"aaaaaa",
    }
    dict4 = {
        "hello1":2,
        "hello2":"2",
    }
    print(f"{dict1 == dict2}")#True
    print(f"{dict1 == dict3}")#False
    print(f"{dict1 == dict4}")#False
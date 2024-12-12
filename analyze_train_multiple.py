import analyze_train
from package.mongodb_handler import MongoHandler
from bson import ObjectId
import os
from datetime import datetime
if __name__ == "__main__":
    
    archive_id_str = input("archive ID: ")
    need_video_rendering_str = input("need Video Rendering(Y/n): ")
    if need_video_rendering_str in ["Y", "y"]:
        need_video_rendering = True
    else:
        need_video_rendering = False
    archive_id = ObjectId(archive_id_str)
    
    mongo_handler = MongoHandler("myosuite_training", "train_archive")
    
    archive_collection = mongo_handler.get_archive_by_id(archive_id)


    print(f"type(archive_collection):{type(archive_collection)}")# dict
    # print(f"archive_collection:{archive_collection}")
    print(f'len session_logs: {len(archive_collection["session_logs"])}')
    print(f'session_logs[0]: {archive_collection["session_logs"][0]}')

    now = datetime.now()
    formatted_timestamp = now.strftime("%Y_%m_%d__%H_%M_%S")
    archive_dir = os.path.join("train_results", f"archive_{formatted_timestamp}_{archive_id}")
    os.makedirs(archive_dir, exist_ok=True)
    for idx, train_session in enumerate(archive_collection["session_logs"]):
        print(f'DEBUG:: {train_session["session_id"]=}')
    for idx, train_session in enumerate(archive_collection["session_logs"]):
        train_session_id = ObjectId(train_session["session_id"])
        print(f'session_id: {train_session_id}')
  
        results_dir = os.path.join(archive_dir, f"{idx:04d}_train_{train_session_id}")
        os.makedirs(results_dir, exist_ok=True)


        analyze_train.plot_result(ObjectId(train_session["session_id"]),
                    result_dir=results_dir,
                    show_train_log_plot=False,
                    flag_watch_trained_sim=False,
                    flag_need_video_rendering=need_video_rendering,
                    object1_idx=-1,
                    )


    # analyze_train.plot_result()
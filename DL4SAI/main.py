
from modules.pipeline import Pipeline

def main():
    conf_out = 0.6
    conf_trs = 1
    conf_dict = {
        0:conf_out, 1:conf_out, 2:conf_out,
        3:conf_trs, 4:conf_trs, 5:conf_trs, 6:conf_trs, 7:conf_trs, 8:conf_trs
    }
    pipeline = Pipeline(verbose=True, use_cached_pcls=False, use_cached_batches=False, data_path="local_ws/MI_front_area",max_image_size=40, densification_mode="NKSR", conf_thres_align=0.5, transition_filter=conf_dict)
    print("Start pipeline")
    pipeline.run()

if __name__ == "__main__":
    main()
import argparse
from src.preprocessing.merge_files import read_files
import os


def f(row):
    topics = [
        "Flu vaccination",
        "Gambling",
        "Online shopping",
        "Social media",
        "cryptocurrency",
        "vegetarianism",
        "violent video games",
        "fossil fuels",
        "doping",
        "privacy laws",
        "autonomous cars",
    ]
    value = row.topic
    index = None
    for item in topics:
        if item in value:
            index = topics.index(item)
            break
    return index


def add_new_topic_column(df):
    tuple_list = df.topic.unique()
    df["new_topic_id"] = df.apply(f, axis=1)
    return df


def main():
    topics = [
        "Flu vaccination",
        "Gambling",
        "Online shopping",
        "Social media",
        "cryptocurrency",
        "vegetarianism",
        "violent video games",
        "fossil fuels",
        "doping",
        "privacy laws",
        "autonomous cars",
    ]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Input Directory",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output Directory"
    )
    args = parser.parse_args()
    df = read_files(args.input_dir)
    df = add_new_topic_column(df)
    for item in topics:
        df_temp = df[df["new_topic_id"] == topics.index(item)]
        df_temp_write = df_temp[["id", "a1", "a2", "label", "topic", "new_topic_id"]]
        df_temp_write.to_csv(os.path.join(args.output_dir, item + ".csv"), index=False)
    df_write = df[["id", "a1", "a2", "label", "topic", "new_topic_id"]]
    df_write.to_csv(os.path.join(args.output_dir, "complete_1.csv"), index=False)


if __name__ == "__main__":
    main()

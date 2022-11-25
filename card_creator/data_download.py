import ijson
import pandas as pd
import requests
import os

data_path = "data/scryfall/card_images/cropped/full/train"


async def get_card_image_urls():
    data = pd.DataFrame()
    response = requests.get(
        "https://archive.scryfall.com/json/scryfall-all-cards.json.gz", stream=True)
    if response.status_code == 200:
        partial_data = []
        for record in ijson.items(response.content, "item"):
            if record["lang"] == "en" and "image_uris" in record:
                partial_data.append(
                    {"id": record["id"], "image_url": record["image_uris"]["art_crop"]})

        data = pd.DataFrame(partial_data)

    else:
        print("failed to fetch the data")

    del response
    return data


async def download_card_images():
    df = get_card_image_urls()
    existing_images = get_existing_images(
        data_path)
    total_count = 0
    new_count = 0
    for i, row in df.iterrows():
        total_count += 1
        try:
            file_name = row[0]
            if (file_name in existing_images):
                continue
            response = requests.get(row[1], stream=True)
            if response.status_code == 200:
                new_count += 1
                with open(f"{data_path}/{file_name}.jpg", 'wb') as f:
                    f.write(response.content)
            else:
                print("failed to fetch the data")
            del response
        except Exception as e:
            print(e)
            print(
                f"====== Failed to fetch the data for {row[0]}. See the error above. ======")

    print(
        f"Attempted {total_count} urls and downloaded {new_count} new images.")


def get_existing_images(path):
    images = []
    for file in os.listdir(path):
        if file.endswith(".gz.jpg"):
            images.append(file[:-7])
    return images


download_card_images()

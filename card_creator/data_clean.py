import ijson
import pandas as pd
import requests
import shutil


def get_card_image_urls():
    with open("data/scryfall/all_cards.json", "rb") as f:
        partial_data = []
        for record in ijson.items(f, "item"):
            if record["lang"] == "en" and "image_uris" in record:
                partial_data.append(
                    {"name": record["name"], "image_url": record["image_uris"]["art_crop"]})

        df = pd.DataFrame(partial_data)
        df.to_csv("data/scryfall/all_card_cropped_images.csv", index=False)


def download_card_images_from_urls():
    df = pd.read_csv("data/scryfall/all_card_cropped_images.csv")
    for i, row in df.iterrows():
        try:
            response = requests.get(row[1], stream=True)
            if response.status_code == 200:
                file_name = row[0].strip().replace(" ", "_")
                for ch in ['\\', '`', '*', '', '{', '}', '[', ']', '(', ')', '>', '#', '+', '-', '.', '!', '$', '\'', '"', '/', ',']:
                    if ch in text:
                        text = text.replace(ch, "")
                with open(f"data/scryfall/card_images/cropped/{file_name}.jpg", 'wb') as out_file:
                    shutil.copyfileobj(response.raw, out_file)
            else:
                print("failed to fetch the data")
            del response
        except Exception as e:
            print(e)
            print(f"failed to fetch the data for {row[0]}")


download_card_images_from_urls()
# get_card_image_urls()

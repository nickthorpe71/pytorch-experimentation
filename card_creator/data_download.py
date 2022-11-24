import ijson
import pandas as pd
import requests
import gzip
import os


def get_card_image_urls():
    with open("data/scryfall/all_cards.json", "rb") as f:
        partial_data = []
        for record in ijson.items(f, "item"):
            if record["lang"] == "en" and "image_uris" in record:
                partial_data.append(
                    {"id": record["id"], "image_url": record["image_uris"]["art_crop"]})

        df = pd.DataFrame(partial_data)
        df.to_csv("data/scryfall/all_card_cropped_images.csv", index=False)


def download_card_images_from_urls():
    df = pd.read_csv("data/scryfall/all_card_cropped_images.csv")
    existing_images = get_existing_images(
        "data/scryfall/card_images/cropped/compressed")
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
                save_as_compressed(
                    response.content, f"data/scryfall/card_images/cropped/compressed/{file_name}.gz.jpg")
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


def format_name(name):
    copy = name[:]
    copy = copy.strip().replace(" ", "_")
    for ch in ['\\', '`', '*', '', '{', '}', '[', ']', '(', ')', '>', '#', '+', '-', '.', '!', '$', '\'', '"', '/', ',']:
        if ch in copy:
            copy = copy.replace(ch, "")
    return copy


def save_as_compressed(data, destination):
    with gzip.open(destination, 'wb') as f:
        f.write(data)


def load_compressed_image(file_name):
    with gzip.open(file_name, 'rb') as f:
        return f.read()


def load_all_compressed_images(path):
    images = []
    for file in os.listdir(path):
        if file.endswith(".gz.jpg"):
            images.append(
                [file[:-7], load_compressed_image(os.path.join(path, file))])
    return images


def save_all_images_as_jpgs(images, path):
    for _, [name, image] in enumerate(images):
        with open(os.path.join(path, f"{name}.jpg"), "wb") as f:
            f.write(image)


# Image size: 626 x 457

# get_card_image_urls()

download_card_images_from_urls()

# save_all_images_as_jpgs(load_all_compressed_images(
#     "data/scryfall/card_images/cropped/compressed"), "data/scryfall/card_images/cropped/full")
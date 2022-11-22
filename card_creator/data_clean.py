import ijson
import pandas as pd

# extract all card image uris from the json file
with open("data/scryfall/all_cards.json", "rb") as f:
    count = 0
    partial_data = []
    for record in ijson.items(f, "item"):
        if record["lang"] == "en" and "image_uris" in record:
            partial_data.append(
                record["image_uris"]["art_crop"])

    df = pd.DataFrame(partial_data)
    df.to_csv("data/scryfall/all_card_cropped_images.csv", index=False)

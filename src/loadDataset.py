from ucimlrepo import fetch_ucirepo
import pandas as pd

print("fetching...")
url_dataset = fetch_ucirepo(id=967)

x = url_dataset.data.features
y = url_dataset.data.targets

df = pd.concat([x,y], axis=1)

output_file = "url_dataset.csv"
df.to_csv(output_file, index=False)

print("Successfully saved")
import json, glob, numpy as np
from geopy.distance import geodesic  # pip if needed, but use tool if no

results = [json.load(open(f)) for f in sorted(glob.glob("result*.json")) if "person" in json.load(open(f))]
# Assume manual true_gps = [(37.7749, -122.4194), ...]  # Your ground truths
true_gps = [(37.7749, -122.4194)] * len(results)  # Placeholder
errors = [geodesic((r["person"]["latitude"], r["person"]["longitude"]), true_gps[i]).meters for i, r in enumerate(results)]
print(f"RMSE: {np.sqrt(np.mean(np.square(errors))):.2f}m | Mean: {np.mean(errors):.2f}m | Max: {np.max(errors):.2f}m")
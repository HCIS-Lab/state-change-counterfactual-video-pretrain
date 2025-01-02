metadata = pd.read_csv(os.path.join('/N/project/ego4d_vlm/egoclip.csv'), sep='\t', on_bad_lines='skip')

with open('/N/project/ego4d_vlm/narration/states.json', "r") as json_file:
    state_metadata = json.load(json_file)
metadata = metadata[metadata['clip_text'].isin(state_metadata.keys())].reset_index(drop=True)

df.to_csv('/N/project/ego4d_vlm/egoclip_update.csv', sep='\t')

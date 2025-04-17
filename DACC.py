import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from collections import Counter
import io
import random

st.set_page_config(layout="wide")
st.title("🟢 Camo Pattern Generator")

# Sidebar Controls
st.sidebar.header("Manipulation Variables")

cutoff_toggle = st.sidebar.toggle("Cutoff")
cutoff_val = st.sidebar.number_input("Cutoff (%)", min_value=0.0, max_value=100.0, value=1.0) if cutoff_toggle else None

layers_toggle = st.sidebar.toggle("Number of Layers")
layers = st.sidebar.number_input("# of Layers", min_value=1, max_value=10, value=3) if layers_toggle else None
light_direction = st.sidebar.radio("Light Direction", ["Back Light", "Forward Light"]) if layers_toggle else None

size_toggle = st.sidebar.toggle("Size")
pixel_block_size = st.sidebar.number_input("Block Size (px)", min_value=1, value=1) if size_toggle else 1

aspect_toggle = st.sidebar.toggle("Aspect Ratio")

round_toggle = st.sidebar.toggle("Round")
round_val = st.sidebar.number_input("Rounding (%)", min_value=0.0, max_value=100.0, value=0.0) if round_toggle else None

# Image Upload & Display
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    image_array = np.array(image)
    total_pixels = image_array.shape[0] * image_array.shape[1]

    if st.button("Analyze"):
        flat_pixels = image_array.reshape(-1, 3)
        hex_codes = ['#{:02x}{:02x}{:02x}'.format(r, g, b) for r, g, b in flat_pixels]
        counter = Counter(hex_codes)
        df = pd.DataFrame(counter.items(), columns=['Hex Code', 'Count'])

        if cutoff_toggle:
            cutoff_count = (cutoff_val / 100.0) * total_pixels
            df = df[df['Count'] >= cutoff_count]

        df['Percentage'] = (df['Count'] / total_pixels) * 100
        df = df.sort_values(by='Count', ascending=False).reset_index(drop=True)
        df['Rank'] = df['Count'].rank(method='min', ascending=False).astype(int)
        df['Use'] = True
        df['Editable %'] = df['Percentage']

        st.session_state['hex_df'] = df
        st.session_state['total_pixels'] = total_pixels
        st.session_state['image_size'] = image_array.shape[:2]

# Hex Code Table
if 'hex_df' in st.session_state:
    st.markdown("### 🎨 Hex Color Analysis")
    df = st.session_state['hex_df']

    edited_df = st.data_editor(
        df[['Use', 'Rank', 'Hex Code', 'Count', 'Percentage', 'Editable %']],
        num_rows="dynamic",
        use_container_width=True,
        key="hex_table_editor"
    )

    df.update(edited_df)
    total_editable = df[df['Use']]['Editable %'].sum()
    if total_editable != 100:
        for i in df.index:
            if df.at[i, 'Use']:
                df.at[i, 'Editable %'] = df.at[i, 'Editable %'] / total_editable * 100

    col1, col2 = st.columns([3,1])
    with col1:
        new_hex = st.text_input("Hex Code", value="#000000")
    with col2:
        if st.button("Add"):
            if new_hex not in df['Hex Code'].values:
                new_row = {
                    'Hex Code': new_hex,
                    'Count': 0,
                    'Percentage': 0.0,
                    'Rank': len(df) + 1,
                    'Use': True,
                    'Editable %': 1.0
                }
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                st.session_state['hex_df'] = df

# Image Generation Logic
if st.button("Generate"):
    df = st.session_state['hex_df']
    h, w = st.session_state['image_size']
    total_pixels = h * w

    palette = df[df['Use']][['Hex Code', 'Editable %']].copy()
    palette['Count'] = (palette['Editable %'] / 100 * total_pixels).astype(int)

    hex_list = []
    for _, row in palette.iterrows():
        hex_list.extend([row['Hex Code']] * row['Count'])

    random.shuffle(hex_list)

    # Handle pixel blocks
    h_blocks = h // pixel_block_size
    w_blocks = w // pixel_block_size
    block_total = h_blocks * w_blocks

    if len(hex_list) < block_total:
        hex_list.extend(random.choices(hex_list, k=block_total - len(hex_list)))
    else:
        hex_list = hex_list[:block_total]

    gen_img = np.zeros((h_blocks, w_blocks, 3), dtype=np.uint8)
    for i in range(h_blocks):
        for j in range(w_blocks):
            hex_color = hex_list[i * w_blocks + j]
            rgb = tuple(int(hex_color[k:k+2], 16) for k in (1, 3, 5))
            gen_img[i, j] = rgb

    # Resize back to original resolution
    result_img = Image.fromarray(gen_img, 'RGB').resize((w, h), Image.NEAREST)
    st.image(result_img, caption="Generated Camo Image", use_column_width=True)
    st.session_state['generated_img'] = result_img

# Download Button
if 'generated_img' in st.session_state:
    buffer = io.BytesIO()
    st.session_state['generated_img'].save(buffer, format="PNG")
    st.download_button(
        label="Download Image",
        data=buffer.getvalue(),
        file_name="camo_pattern.png",
        mime="image/png"
    )

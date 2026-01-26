import pandas as pd
import streamlit as st

@st.cache_data
def load_data(file):
    try:
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        else:
            return pd.read_excel(file)
    except Exception as e:
        return None

# --- THEMES ---
journal_themes = {
    "Nature (NPG)": ["#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F", "#8491B4", "#91D1C2", "#DC0000", "#7E6148", "#B09C85"],
    "Science (AAAS)": ["#3B4992", "#EE0000", "#008B45", "#631879", "#008280", "#BB0021", "#5F559B", "#A20056", "#808180", "#1B1919"],
    "JCO": ["#0073C2", "#EFC000", "#868686", "#CD534C", "#7AA6DC", "#003C67", "#8F7700", "#3B3B3B", "#A73030", "#4A6990"],
    "Lancet": ["#00468B", "#ED0000", "#42B540", "#0099B4", "#925E9F", "#FDAF91", "#AD002A", "#ADB6B6", "#1B1919"],
    "NEJM": ["#BC3C29", "#0072B5", "#E18727", "#20854E", "#7876B1", "#6F99AD", "#FFDC91", "#EE4C97"],
    "Blood": ["#AA0000", "#E00000", "#8B0000", "#FF0000", "#B22222", "#FF69B4", "#800000"], 
    "Leukemia": ["#377EB8", "#E41A1C", "#4DAF4A", "#984EA3", "#FF7F00", "#FFFF33", "#A65628", "#F781BF"], 
    "Jama": ["#374E55", "#DF8F44", "#00A1D5", "#B24745", "#79AF97", "#6A6599", "#80796B"],
}

fun_themes = {
    "Cosmic Nihilism": ["#A6EEE6", "#F0F035", "#44281D", "#E4A71B", "#8BCF21", "#FBFBFB"],
    "Hollywood Equine": ["#2C3E50", "#D35400", "#2980B9", "#C0392B", "#bdc3c7", "#F39C12"],
    "Prehistoric One": ["#5D4037", "#D84315", "#388E3C", "#FBC02D", "#455A64", "#212121"],
    "Alien Biosphere": ["#FFB7E1", "#8DE581", "#7DFAFF", "#D78E38", "#596025", "#304245"], # Scavengers Reign
    "Alien Flora": ["#F4EBD0", "#D66853", "#3C505D", "#7D9D9C", "#212D40", "#A8C686"],
    "Neon Acid": ["#FF00FF", "#00FF00", "#0000FF", "#FFFF00", "#00FFFF", "#FF0000"],
    "Cyber Grid": ["#2F5061", "#4291C7", "#D57FBE", "#E45D5C", "#FFAE91", "#F9DB57", "#FFFFD0"],
    "Dream Layers": ["#EFAC3D", "#842650", "#69932B", "#5F2D7B", "#5C5D5E"],
    "Deep Space": ["#0B3D91", "#F2C45A", "#465362", "#A2BCE0", "#1E1E24"],
    "Office Separation": ["#F0F6F7", "#89AEC8", "#7B6727", "#4E452A", "#002C55"],
    "Sudden Departure": ["#F2A78C", "#F2DCBB", "#B4B4BB", "#77708C", "#C997A2"],
    "Family Empire": ["#1C2541", "#3A506B", "#5BC0BE", "#6FFFE9", "#0B132B"],
    "Practice Run": ["#e0e1dd", "#778da9", "#415a77", "#1b263b", "#0d1b2a"],
    "Exclusion Zone": ["#D8FF00", "#E0FF33", "#333333", "#555555", "#D65050"],
    "Seven Kingdoms": ["#808080", "#FFD700", "#B22222", "#000000", "#228B22"],
    "Indian Train Journey": ["#FF0000", "#00A08A", "#F2AD00", "#F98400", "#5BBCD6"],
    "Grand Hotel": ["#F1BB7B", "#FD6467", "#5B1A18", "#D67236"],
    "Gotham Night": ["#0C2340", "#282D3C", "#808080", "#000000", "#710193", "#32CD32", "#B3B3B3"],
    "Coal Town Saga": ["#212121", "#B71C1C", "#FFC107", "#5D4037", "#00C853", "#E65100"],
    "Tragedy of a King": ["#FFD700", "#DC143C", "#0000CD", "#DAA520", "#000000", "#FFFFFF"],
    "Monochrome on the Road": ["#1A1A1A", "#4D4D4D", "#808080", "#B3B3B3", "#E6E6E6", "#F0EAD6"], 
    "Anime Fantasy": ["#8CBF88", "#E53935", "#607D8B", "#FFA500", "#D2E3EF", "#FF6347"],

    # New Cult Classics
    "Manic Urbanism": ["#E91E63", "#D32F2F", "#607D8B", "#455A64", "#212121"], 
    "Idol's Nightmare": ["#FF69B4", "#D50000", "#1A237E", "#F0F8FF", "#880E4F", "#000000"], # Perfect Blue 
    "Nameless Terror": ["#5D4037", "#3E2723", "#B71C1C", "#263238", "#ECEFF1"], 
    "The Great Epic": ["#FF9800", "#FFC107", "#D32F2F", "#00BCD4", "#795548"], 
    "Wizard of Loneliness": ["#E0E0E0", "#90A4AE", "#546E7A", "#A1887F", "#B0BEC5"], 
    
    # Literary Themes
    "Saint Petersburg 1866": ["#3E2723", "#BF360C", "#F9A825", "#424242", "#ECEFF1"], 
    "The Law Clerk": ["#263238", "#546E7A", "#78909C", "#D7CCC8", "#8D6E63", "#212121"], 
    "Stream of Life": ["#D81B60", "#F48FB1", "#4A148C", "#FFF176", "#00BCD4"], 
    "Cosmic Ocean": ["#311B92", "#7C4DFF", "#00E676", "#3E2723", "#FF6F00"], 
    "The Playwright": ["#607D8B", "#8D6E63", "#CFD8DC", "#A1887F", "#546E7A"], 
}

all_themes = {**journal_themes, **fun_themes}

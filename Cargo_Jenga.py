import streamlit as st
import pandas as pd
import itertools

# ----------------- Predefined Containers -----------------
containers = {
    "CJ": {
        "door": {"width_min": 24, "width_max": 26, "height": 20, "diag": 31},
        "interior": {"height": 22, "depth": 45, "width": 84, "restricted": {"width": 20, "depth": 20}}
    },
    "Legacy": {
        "door": {"width": 34, "height": 22, "diag": 38},
        "interior": {"height": 36, "depth_min": 36, "depth_max": 54, "width_min": 36, "width_max": 89}
    }
}

# ----------------- Standard Baggage Presets -----------------
standard_baggage = {
    "Small Carry-on": (22, 14, 9),
    "Medium Suitcase": (24, 16, 10),
    "Large Suitcase": (28, 20, 12),
    "Golf Clubs": (50, 15, 12),
    "Ski Bag": (72, 10, 7),
    "Custom": None
}

# ----------------- Helper Functions -----------------
def fits_through_door(box_dims, door):
    l, w, h = box_dims
    for dims in itertools.permutations([l, w, h]):
        bw, bh = dims[0], dims[1]  # orientation
        diag = (bw**2 + bh**2) ** 0.5
        if "width_min" in door:  # CJ
            if (bw <= door["width_max"] and bh <= door["height"]) or diag <= door["diag"]:
                return True
        else:  # Legacy
            if (bw <= door["width"] and bh <= door["height"]) or diag <= door["diag"]:
                return True
    return False

def fits_inside(box_dims, interior, container_type):
    l, w, h = box_dims
    for dims in itertools.permutations([l, w, h]):
        bl, bw, bh = dims
        if container_type == "CJ":
            if bh <= interior["height"] and bl <= interior["depth"] and bw <= interior["width"]:
                return True
        elif container_type == "Legacy":
            if bh <= interior["height"] and bl <= interior["depth_max"] and bw <= interior["width_max"]:
                return True
    return False

# ----------------- Streamlit UI -----------------
st.title("Aircraft Cargo Fit Checker")

container_choice = st.selectbox("Select Aircraft Cargo Hold", list(containers.keys()))
container = containers[container_choice]

if "baggage_list" not in st.session_state:
    st.session_state["baggage_list"] = []

st.write("### Add Baggage")
baggage_type = st.selectbox("Baggage Type", list(standard_baggage.keys()))

if baggage_type == "Custom":
    length = st.number_input("Length (in)", min_value=1)
    width = st.number_input("Width (in)", min_value=1)
    height = st.number_input("Height (in)", min_value=1)
    dims = (length, width, height)
else:
    dims = standard_baggage[baggage_type]

qty = st.number_input("Quantity", min_value=1, value=1)

if st.button("Add Item"):
    if dims is None:
        st.warning("Please enter dimensions for custom item.")
    else:
        for _ in range(qty):
            st.session_state["baggage_list"].append({"Type": baggage_type, "Dims": dims})
        st.success(f"Added {qty} × {baggage_type}")

# Show current baggage list
if st.session_state["baggage_list"]:
    df = pd.DataFrame(st.session_state["baggage_list"])
    st.write("### Current Baggage Load")
    st.table(df)

    if st.button("Check Fit"):
        results = []
        for i, item in enumerate(st.session_state["baggage_list"], 1):
            box_dims = item["Dims"]
            door_fit = fits_through_door(box_dims, container["door"])
            interior_fit = fits_inside(box_dims, container["interior"], container_choice)
            if door_fit and interior_fit:
                status = "✅ Fits"
            elif not door_fit:
                status = "❌ Door Fail"
            else:
                status = "❌ Interior Fail"
            results.append({"Item": i, "Type": item["Type"], "Dims": box_dims, "Result": status})

        st.write("### Fit Results")
        st.table(pd.DataFrame(results))

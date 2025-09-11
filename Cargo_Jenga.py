import streamlit as st
import pandas as pd
import itertools
import plotly.graph_objects as go

# ----------------- Predefined Containers -----------------
containers = {
    "CJ": {
        "door": {"width_min": 24, "width_max": 26, "height": 20, "diag": 31},
        "interior": {
            "height": 22,
            "depth": 45,
            "width": 84,
            "restricted": {"width": 20, "depth": 20}
        }
    },
    "Legacy": {
        "door": {"width": 34, "height": 22, "diag": 38},
        "interior": {
            "height": 36,
            "depth": 89,
            "width_min": 36,
            "width_max": 54
        }
    }
}

# ----------------- Standard Baggage Presets (with Flexibility) -----------------
standard_baggage = {
    "Small Carry-on": {"dims": (22, 14, 9), "flex": 1.0},
    "Standard Suitcase": {"dims": (26, 18, 10), "flex": 1.0},
    "Large Suitcase": {"dims": (30, 19, 11), "flex": 1.0},
    "Golf Clubs (Hard Case)": {"dims": (50, 14, 14), "flex": 1.0},
    "Golf Clubs (Soft Bag)": {"dims": (50, 14, 14), "flex": 0.9},  # tweak here if desired
    "Ski Bag (Soft)": {"dims": (70, 12, 7), "flex": 0.9},
    "Custom": {"dims": None, "flex": 1.0}
}

# ----------------- Helpers -----------------
def fits_through_door(box_dims, door):
    l, w, h = box_dims
    for bw, bh, _ in itertools.permutations([l, w, h]):
        diag = (bw**2 + bh**2) ** 0.5
        if "width_min" in door:  # CJ
            if (bw <= door["width_max"] and bh <= door["height"]) or diag <= door["diag"]:
                return True
        else:  # Legacy
            if (bw <= door["width"] and bh <= door["height"]) or diag <= door["diag"]:
                return True
    return False

def legacy_width_at_height(interior, z):
    h = interior["height"]
    wmin, wmax = interior["width_min"], interior["width_max"]
    return wmin + (wmax - wmin) * (z / h)

def apply_flex(dims, flex):
    l, w, h = dims
    return (l * flex, w * flex, h * flex)

def fits_inside(box_dims, interior, container_type, flex=1.0):
    l, w, h = apply_flex(box_dims, flex)
    for bl, bw, bh in itertools.permutations([l, w, h]):
        if container_type == "CJ":
            if bh <= interior["height"] and bl <= interior["depth"] and bw <= interior["width"]:
                return True
        elif container_type == "Legacy":
            if bl <= interior["depth"] and bh <= interior["height"]:
                if bw <= min(legacy_width_at_height(interior, 0),
                             legacy_width_at_height(interior, bh)):
                    return True
    return False

# AABB overlap
def aabb_overlap(a, b):
    ax0, ay0, az0, ax1, ay1, az1 = a
    bx0, by0, bz0, bx1, by1, bz1 = b
    return not (ax1 <= bx0 or ax0 >= bx1 or
                ay1 <= by0 or ay0 >= by1 or
                az1 <= bz0 or az0 >= bz1)

def point_in_cj_restricted(x, y, interior, cargo_W):
    r = interior["restricted"]
    rx0, rx1 = 0, r["depth"]
    ry0, ry1 = cargo_W - r["width"], cargo_W
    return (rx0 <= x <= rx1) and (ry0 <= y <= ry1)

def within_bounds_and_constraints(x0, y0, z0, l, w, h, container_type, interior, cargo_dims):
    L, W, H = cargo_dims
    x1, y1, z1 = x0 + l, y0 + w, z0 + h
    # global bounds
    if x0 < 0 or y0 < 0 or z0 < 0: return False
    if x1 > L or z1 > H: return False

    if container_type == "CJ":
        if y1 > W: return False
        # restricted block (full height)
        r = interior["restricted"]
        rx0, rx1 = 0, r["depth"]
        ry0, ry1 = W - r["width"], W
        rz0, rz1 = 0, interior["height"]
        overlap = not (x1 <= rx0 or x0 >= rx1 or
                       y1 <= ry0 or y0 >= ry1 or
                       z1 <= rz0 or z0 >= rz1)
        if overlap: return False
        return True

    else:  # Legacy taper — enforce width at bottom & top of the box
        w_bottom = legacy_width_at_height(interior, z0)
        w_top = legacy_width_at_height(interior, z1)
        w_avail = min(w_bottom, w_top)
        return y1 <= w_avail

def dedupe_points(points):
    seen = set()
    out = []
    for p in points:
        key = (round(p[0], 4), round(p[1], 4), round(p[2], 4))
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out

def prune_dominated_points(points):
    # Remove points that are dominated (another point <= in all axes)
    pruned = []
    for i, p in enumerate(points):
        dominated = False
        for j, q in enumerate(points):
            if i == j: continue
            if q[0] <= p[0] and q[1] <= p[1] and q[2] <= p[2] and (q[0] < p[0] or q[1] < p[1] or q[2] < p[2]):
                dominated = True
                break
        if not dominated:
            pruned.append(p)
    return pruned

def get_orientations(dims):
    # unique orientations
    return set(itertools.permutations(dims, 3))

# ----------------- Extreme-Points 3D Packer -----------------
def extreme_points_packing(baggage_list, container_type, interior):
    # Container dims for placement
    if container_type == "CJ":
        L, W, H = interior["depth"], interior["width"], interior["height"]
    else:  # Legacy: use depth; width enforced per z by constraint
        L, W, H = interior["depth"], interior["width_max"], interior["height"]

    cargo_dims = (L, W, H)

    # Sort items by "difficulty": larger flexed max-dimension first, then volume
    sortable = []
    for idx, item in enumerate(baggage_list):
        dims_f = apply_flex(item["Dims"], item.get("Flex", 1.0))
        volume = dims_f[0] * dims_f[1] * dims_f[2]
        sortable.append((max(dims_f), volume, idx, item))
    sortable.sort(key=lambda t: (-t[0], -t[1]))

    # Extreme points set starts at origin
    points = [(0.0, 0.0, 0.0)]
    placed_boxes = []  # store AABBs for collision
    placements = []    # store user-visible placements (real dims)

    for _, __, _, item in sortable:
        dims_real = item["Dims"]
        dims_flex = apply_flex(dims_real, item.get("Flex", 1.0))
        best = None  # (score, (x0,y0,z0,l,w,h), oriented_dims_flex)

        # try all extreme points + orientations
        # sort points to prefer lowest z, then y, then x
        for (px, py, pz) in sorted(points, key=lambda p: (p[2], p[1], p[0])):
            # skip points outside feasible regions
            if container_type == "CJ":
                # Don't start inside restricted block footprint
                if point_in_cj_restricted(px, py, interior, W):
                    continue
            else:
                # Legacy: point must be inside current width at its height
                if py > legacy_width_at_height(interior, pz):
                    continue

            for (l, w, h) in get_orientations(dims_flex):
                x0, y0, z0 = px, py, pz
                # bounds / constraint check
                if not within_bounds_and_constraints(x0, y0, z0, l, w, h, container_type, interior, cargo_dims):
                    continue
                # collision check with placed boxes (use flexed dims for clearance)
                candidate = (x0, y0, z0, x0 + l, y0 + w, z0 + h)
                collide = any(aabb_overlap(candidate, b) for b in placed_boxes)
                if collide:
                    continue

                # scoring heuristic: prefer lowest top (z1), then low y1, then low x1
                z1, y1, x1 = candidate[5], candidate[4], candidate[3]
                score = (z1, y1, x1)

                if (best is None) or (score < best[0]):
                    best = (score, (x0, y0, z0, l, w, h))

        if best is None:
            # could not place this item
            return False, placements

        # commit placement
        _, (x0, y0, z0, l, w, h) = best
        placed_boxes.append((x0, y0, z0, x0 + l, y0 + w, z0 + h))
        placements.append({
            "Item": len(placements) + 1,
            "Type": item["Type"],
            "Dims": dims_real,           # show true size
            "Position": (x0, y0, z0)     # placed at this origin
        })

        # generate new extreme points: right, front, above of this box
        new_pts = [
            (x0 + l, y0, z0),
            (x0, y0 + w, z0),
            (x0, y0, z0 + h),
        ]

        # filter infeasible points early
        filt = []
        for (nx, ny, nz) in new_pts:
            # global bounds quick check
            if nx > L or nz > H:
                continue
            if container_type == "CJ":
                if ny > W:  # outside width
                    continue
                # reject points inside restricted footprint
                if point_in_cj_restricted(nx, ny, interior, W):
                    continue
            else:
                # Legacy: point must lie within taper width at its height
                if ny > legacy_width_at_height(interior, nz):
                    continue
            # also skip points that are inside any already placed box
            inside_existing = False
            for bx in placed_boxes:
                if (bx[0] <= nx <= bx[3]) and (bx[1] <= ny <= bx[4]) and (bx[2] <= nz <= bx[5]):
                    inside_existing = True
                    break
            if not inside_existing:
                filt.append((nx, ny, nz))

        points.extend(filt)
        points = dedupe_points(points)
        points = prune_dominated_points(points)

    return True, placements

# ----------------- Visualization -----------------
def plot_cargo(cargo_dims, placements, container_type=None, interior=None):
    cargo_L, cargo_W, cargo_H = cargo_dims
    fig = go.Figure()

    # Cargo hold for CJ (edges)
    if container_type == "CJ":
        corners = [
            (0,0,0), (cargo_L,0,0), (cargo_L,cargo_W,0), (0,cargo_W,0),
            (0,0,cargo_H), (cargo_L,0,cargo_H), (cargo_L,cargo_W,cargo_H), (0,cargo_W,cargo_H)
        ]
        edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
        for e in edges:
            x = [corners[e[0]][0], corners[e[1]][0]]
            y = [corners[e[0]][1], corners[e[1]][1]]
            z = [corners[e[0]][2], corners[e[1]][2]]
            fig.add_trace(go.Scatter3d(x=x,y=y,z=z,mode='lines',
                                       line=dict(color='black',width=4),
                                       name='Cargo Hold'))

        # CJ restricted block
        r = interior["restricted"]
        x0, y0, z0 = 0, cargo_W - r["width"], 0
        x1, y1, z1 = r["depth"], cargo_W, interior["height"]
        vertices = [
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]
        ]
        x, y, z = zip(*vertices)
        faces = [(0,1,2),(0,2,3),(4,5,6),(4,6,7),
                 (0,1,5),(0,5,4),(1,2,6),(1,6,5),
                 (2,3,7),(2,7,6),(3,0,4),(3,4,7)]
        i, j, k = zip(*faces)
        fig.add_trace(go.Mesh3d(x=x,y=y,z=z,i=i,j=j,k=k,
                                color='gray', opacity=0.4,
                                name='Restricted Area'))

    # Cargo hold for Legacy (rectangular prism with taper)
    if container_type == "Legacy":
        d = interior["depth"]
        wmin, wmax = interior["width_min"], interior["width_max"]
        h = interior["height"]
        vertices = [
            [0, 0, 0], [d, 0, 0], [d, wmin, 0], [0, wmin, 0],
            [0, 0, h], [d, 0, h], [d, wmax, h], [0, wmax, h]
        ]
        x, y, z = zip(*vertices)
        faces = [(0,1,2),(0,2,3),(4,5,6),(4,6,7),
                 (0,1,5),(0,5,4),(1,2,6),(1,6,5),
                 (2,3,7),(2,7,6),(3,0,4),(3,4,7)]
        i, j, k = zip(*faces)
        fig.add_trace(go.Mesh3d(x=x,y=y,z=z,i=i,j=j,k=k,
                                color='lightblue', opacity=0.15,
                                name='Legacy Cargo Hold'))

    # Add baggage (true sizes)
    colors = ['red','green','blue','orange','purple','cyan','magenta']
    for idx, item in enumerate(placements):
        l, w, h = item["Dims"]
        x0, y0, z0 = item["Position"]
        color = colors[idx % len(colors)]
        x = [x0, x0+l, x0+l, x0, x0, x0+l, x0+l, x0]
        y = [y0, y0, y0+w, y0+w, y0, y0, y0+w, y0+w]
        z = [z0, z0, z0, z0, z0+h, z0+h, z0+h, z0+h]
        fig.add_trace(go.Mesh3d(x=x,y=y,z=z,color=color,opacity=0.5,name=item["Type"]))

    fig.update_layout(
        scene=dict(
            xaxis_title='Depth (in)', yaxis_title='Width (in)', zaxis_title='Height (in)',
            aspectmode='data'
        ),
        margin=dict(l=0,r=0,b=0,t=0)
    )
    return fig

# ----------------- Streamlit UI -----------------
st.title("Aircraft Cargo Fit Checker — Extreme-Points Packing")

container_choice = st.selectbox("Select Aircraft Cargo Hold", list(containers.keys()))
container = containers[container_choice]

if "baggage_list" not in st.session_state:
    st.session_state["baggage_list"] = []

if st.button("Clear Baggage List"):
    st.session_state["baggage_list"] = []
    st.success("✅ Baggage list cleared.")

st.write("### Add Baggage")
baggage_type = st.selectbox("Baggage Type", list(standard_baggage.keys()))

if baggage_type == "Custom":
    length = st.number_input("Length (in)", min_value=1)
    width = st.number_input("Width (in)", min_value=1)
    height = st.number_input("Height (in)", min_value=1)
    dims = (length, width, height)
    flex = 1.0
else:
    dims = standard_baggage[baggage_type]["dims"]
    flex = standard_baggage[baggage_type]["flex"]

qty = st.number_input("Quantity", min_value=1, value=1)

if st.button("Add Item"):
    if dims is None:
        st.warning("Please enter dimensions for custom item.")
    else:
        for _ in range(qty):
            st.session_state["baggage_list"].append({
                "Type": baggage_type,
                "Dims": dims,
                "Flex": flex
            })
        st.success(f"Added {qty} × {baggage_type}")

if st.session_state["baggage_list"]:
    df = pd.DataFrame(st.session_state["baggage_list"]).reset_index(drop=True)
    df.index = df.index + 1
    df.index.name = "Item"
    st.write("### Current Baggage Load")
    st.table(df)

    if st.button("Check Fit"):
        # Per-item checks
        results = []
        for item in st.session_state["baggage_list"]:
            box_dims = item["Dims"]
            door_fit = fits_through_door(box_dims, container["door"])
            interior_fit = fits_inside(box_dims, container["interior"], container_choice, item.get("Flex", 1.0))
            status = "✅ Fits" if door_fit and interior_fit else "❌ Door Fail" if not door_fit else "❌ Interior Fail"
            results.append({"Type": item["Type"], "Dims": box_dims, "Result": status})

        results_df = pd.DataFrame(results).reset_index(drop=True)
        results_df.index = results_df.index + 1
        results_df.index.name = "Item"
        st.write("### Fit Results")
        st.table(results_df)

        # Extreme-Points overall packing
        success, placements = extreme_points_packing(
            st.session_state["baggage_list"], container_choice, container["interior"]
        )
        st.write("### Overall Cargo Packing Feasibility (Extreme-Points)")
        if success:
            st.success("✅ Packing possible.")
        else:
            st.error("❌ Packing failed.")

        if placements:
            placements_df = pd.DataFrame(placements).reset_index(drop=True)
            placements_df.index = placements_df.index + 1
            placements_df.index.name = "Item"
            st.write("### Suggested Placement Positions")
            st.table(placements_df)

            st.write("### Cargo Load Visualization")
            if container_choice == "CJ":
                cargo_dims = (container["interior"]["depth"],
                              container["interior"]["width"],
                              container["interior"]["height"])
            else:
                cargo_dims = (container["interior"]["depth"],
                              container["interior"]["width_max"],
                              container["interior"]["height"])
            fig = plot_cargo(cargo_dims, placements, container_choice, container["interior"])
            st.plotly_chart(fig, use_container_width=True)

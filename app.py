import streamlit as st

# ── Data Loader ───────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    urls = [
        "https://raw.githubusercontent.com/dsrscientist/dataset1/master/superstore.csv",
        "https://raw.githubusercontent.com/FlipRoboTechnologies/ML_-Data-Sets/main/First%20Semester/Superstore.csv",
    ]

    df = None

    for url in urls:
        try:
            df = pd.read_csv(url, encoding="latin-1")
            break
        except Exception:
            continue

    # ── Fallback Synthetic Dataset ────────────────────────────────
    if df is None:
        np.random.seed(42)
        n = 9994

        regions = ["West", "East", "Central", "South"]
        cats = ["Technology", "Furniture", "Office Supplies"]

        sub_cats = {
            "Technology": ["Phones", "Computers", "Accessories", "Copiers"],
            "Furniture": ["Chairs", "Tables", "Bookcases", "Furnishings"],
            "Office Supplies": ["Paper", "Binders", "Storage", "Art", "Labels", "Fasteners"]
        }

        segs = ["Consumer", "Corporate", "Home Office"]
        ships = ["Standard Class", "Second Class", "First Class", "Same Day"]
        custs = [f"Customer_{i}" for i in range(1, 794)]

        cat_arr = np.random.choice(cats, n)
        sub_arr = [np.random.choice(sub_cats[c]) for c in cat_arr]

        dates = pd.date_range("2019-01-01", "2022-12-31", periods=n)

        sales = np.round(np.random.exponential(250, n) + 10, 2)
        disc = np.round(
            np.random.choice([0, .1, .2, .3, .4, .5], n,
                             p=[.5, .15, .15, .1, .05, .05]), 2
        )

        profit = np.round(
            sales * (0.2 - disc + np.random.normal(0, .05, n)), 2
        )

        df = pd.DataFrame({
            "Row ID": range(1, n + 1),
            "Order ID": [
                f"CA-{np.random.randint(2019,2023)}-{np.random.randint(100000,999999)}"
                for _ in range(n)
            ],
            "Order Date": dates,
            "Ship Date": dates + pd.to_timedelta(np.random.randint(1, 8, n), "D"),
            "Ship Mode": np.random.choice(ships, n),
            "Customer ID": [f"CU-{np.random.randint(1000,9999)}" for _ in range(n)],
            "Customer Name": np.random.choice(custs, n),
            "Segment": np.random.choice(segs, n),
            "Country": "United States",
            "City": np.random.choice(
                ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"], n
            ),
            "State": np.random.choice(
                ["California", "New York", "Texas", "Illinois", "Arizona"], n
            ),
            "Region": np.random.choice(regions, n),
            "Category": cat_arr,
            "Sub-Category": sub_arr,
            "Sales": sales,
            "Quantity": np.random.randint(1, 15, n),
            "Discount": disc,
            "Profit": profit,
        })

    # ── Column Cleanup ────────────────────────────────────────────
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    # ── Datetime Conversion (FIXED) ──────────────────────────────
    for c in ["Order_Date", "Ship_Date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    # ── Numeric Conversion ───────────────────────────────────────
    for c in ["Sales", "Profit", "Quantity", "Discount"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # ── Remove Missing/Duplicates ────────────────────────────────
    df.dropna(subset=["Sales", "Profit"], inplace=True)
    df.drop_duplicates(inplace=True)

    # ── Date Features ────────────────────────────────────────────
    if "Order_Date" in df.columns:
        df["Year"] = df["Order_Date"].dt.year
        df["Quarter"] = df["Order_Date"].dt.to_period("Q").astype(str)
        df["YearMonth"] = df["Order_Date"].dt.to_period("M").astype(str)

    # ── Delivery Days ────────────────────────────────────────────
    if "Order_Date" in df.columns and "Ship_Date" in df.columns:
        df["Delivery_Days"] = (
            df["Ship_Date"] - df["Order_Date"]
        ).dt.days

    # ── Profit Margin ────────────────────────────────────────────
    df["Profit_Margin"] = np.where(
        df["Sales"] > 0,
        (df["Profit"] / df["Sales"]) * 100,
        0
    ).round(2)

    # ── Text Cleanup ─────────────────────────────────────────────
    for c in ["Region", "Category", "Sub_Category", "Segment", "Ship_Mode"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.title()

    return df
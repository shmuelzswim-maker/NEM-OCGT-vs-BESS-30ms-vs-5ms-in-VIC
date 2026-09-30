import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _():
    # ── CELL 1: IMPORTS & CONFIGURATION ───────────────────────────────────────────

    import marimo as mo

    import nemosis

    from nemosis import dynamic_data_compiler, static_table

    import pandas as pd

    import numpy as np

    import matplotlib.pyplot as plt

    import matplotlib.ticker as ticker

    import matplotlib.patches as mpatches

    import os

    import requests

    import warnings

    warnings.filterwarnings('ignore')


    CACHE_DIR  = './nemosis_cache'

    VIC_REGION = 'VIC1'

    os.makedirs(CACHE_DIR, exist_ok=True)


    # Bypass AEMO bot-blocker for the registry file

    excel_file = os.path.join(CACHE_DIR, 'NEM Registration and Exemption List.xls')

    if not os.path.exists(excel_file):

        print("Downloading AEMO registry manually...")

        url = 'https://aemo.com.au/-/media/files/electricity/nem/participant_information/nem-registration-and-exemption-list.xls'

        headers = {'User-Agent': 'Mozilla/5.0'}

        response = requests.get(url, headers=headers)

        with open(excel_file, 'wb') as f:

            f.write(response.content)


    # ── ANALYTICAL WINDOWS ────────────────────────────────────────────────────────

    PRE_START  = '2019/10/01 00:00:00'

    PRE_END    = '2021/10/01 00:00:00'

    POST_START = '2021/10/01 00:00:00'

    POST_END   = '2024/10/01 00:00:00'


    # Three sub-periods for transparent presentation (no data is hidden)

    # Period A: Pre-5MS baseline

    PERIOD_A_START = pd.to_datetime('2019-10-01')

    PERIOD_A_END   = pd.to_datetime('2021-10-01')


    # Period B: Post-5MS early / crisis-affected

    # Gas crisis generally dated from AEMO market suspension in June 2022.

    # We begin Period B at 5MS commencement and end it at Dec 2022 to capture

    # the full crisis tail. This period is FLAGGED, not excised.

    PERIOD_B_START = pd.to_datetime('2021-10-01')

    PERIOD_B_END   = pd.to_datetime('2022-12-31')


    # Period C: Post-5MS normal (2023 onward — crisis effects dissipated)

    PERIOD_C_START = pd.to_datetime('2023-01-01')

    PERIOD_C_END   = pd.to_datetime('2024-10-01')


    mo.md("""

    # 5MS Impact Analysis

    ### Dispatch-Weighted Price (DWP) Comparison: VIC Batteries vs OCGTs (2019–2024)


    **Research question:** Did the transition to 5-Minute Settlement shift

    dispatch-weighted prices in favour of fast-response technologies?


    **Methodology:**

    - Price capture efficiency (DWP) is the sole metric — no revenue calculations.

    - Battery load (charging) DUIDs are excluded; only active discharge is measured.

    - Rather than excising the 2022 energy crisis, the post-5MS period is split into

      three transparent sub-periods so crisis-affected data is visible but clearly labelled.

    - OCGT cohort requires ≥2,000 active dispatch intervals per period for stability.

    - Battery fleet bypasses the stability filter (emerging technology, post-2021 commissioning).


    **Periods:**

    | Label | Date Range | Notes |

    |---|---|---|

    | Pre-5MS | Oct 2019 – Oct 2021 | 30-min settlement baseline |

    | Post-5MS Early | Oct 2021 – Dec 2022 | Crisis-affected, flagged |

    | Post-5MS Normal | Jan 2023 – Oct 2024 | Post-crisis, clean comparison |

    """)
    return (
        CACHE_DIR,
        PERIOD_A_END,
        PERIOD_A_START,
        PERIOD_B_END,
        PERIOD_B_START,
        PERIOD_C_END,
        PERIOD_C_START,
        POST_END,
        POST_START,
        PRE_END,
        PRE_START,
        VIC_REGION,
        dynamic_data_compiler,
        mo,
        np,
        os,
        pd,
        plt,
        static_table,
        ticker,
    )


@app.cell
def _(CACHE_DIR, mo, os, pd, static_table):
    # ── CELL 2: FLEET DUID IDENTIFICATION ─────────────────────────────────────────

    mo.md("## Step 1: Identify Fleet DUIDs (Discharge Only)")


    generators = static_table(table_name='Generators and Scheduled Loads', raw_data_location=CACHE_DIR)

    generators.columns = generators.columns.str.strip()

    vic_gen = generators[generators['Region'].astype(str).str.upper().str.contains('VIC', na=False)].copy()


    # Identify Battery Generators (Discharge only)

    battery_mask = (

        vic_gen['Fuel Source - Primary'].astype(str).str.upper().str.contains('BATTERY|STORAGE', na=False) |

        vic_gen['Technology Type - Primary'].astype(str).str.upper().str.contains('BATTERY|STORAGE', na=False) |

        vic_gen['Technology Type - Descriptor'].astype(str).str.upper().str.contains('BATTERY|STORAGE', na=False)

    )

    vic_batteries = vic_gen[battery_mask].copy()

    current_bdus = vic_batteries['DUID'].astype(str).str.strip().unique().tolist()


    # Manually inject pre-2024 'G1' generator DUIDs (before AEMO BDU reclassification)

    historical_gens = ['BALBG1', 'GANNBG1', 'VBBG1', 'BULBG1', 'HBESG1', 'KESSG1', 'PIBESG1']

    battery_gen_duids = list(set(current_bdus + historical_gens))


    # Identify OCGTs

    ocgt_mask = (

        vic_gen['Technology Type - Primary'].astype(str).str.upper().str.contains('OCGT|OPEN CYCLE|COMBUSTION|GAS TURBINE', na=False) |

        vic_gen['Technology Type - Descriptor'].astype(str).str.upper().str.contains('OCGT|OPEN CYCLE|COMBUSTION|GAS TURBINE', na=False)

    ) & vic_gen['Fuel Source - Primary'].astype(str).str.upper().str.contains('GAS|LIQUID|FOSSIL', na=False)


    vic_ocgt = vic_gen[ocgt_mask].copy()

    coal_steam_duids = ['LYA1', 'LYA2', 'LYA3', 'LYA4', 'LOYYB1', 'LOYYB2',

                        'YWPS1', 'YWPS2', 'YWPS3', 'YWPS4', 'NPS']

    vic_ocgt = vic_ocgt[~vic_ocgt['DUID'].isin(coal_steam_duids)]

    ocgt_duids = vic_ocgt['DUID'].astype(str).str.strip().unique().tolist()


    # ── Build capacity map from AEMO registry ────────────────────────────────────

    # NEMOSIS's static_table loader only pulls a fixed subset of columns from the

    # NEM Registration and Exemption List and does not include 'Reg Cap generation (MW)'.

    # We therefore read the 'PU and Scheduled Loads' sheet directly from the cached

    # Excel file, which contains the full column set including registered capacities.

    # Source: AEMO NEM Registration and Exemption List, sheet 'PU and Scheduled Loads'.

    registry_path = os.path.join(CACHE_DIR, 'NEM Registration and Exemption List.xls')

    full_registry = pd.read_excel(registry_path, sheet_name='PU and Scheduled Loads')

    full_registry.columns = full_registry.columns.str.strip()


    cap_col = 'Reg Cap generation (MW)'

    print(f"Registry sheet columns: {full_registry.columns.tolist()}")

    print(f"Using capacity column:  '{cap_col}'")


    all_relevant_duids = battery_gen_duids + ocgt_duids

    cap_df = full_registry[full_registry['DUID'].isin(all_relevant_duids)][['DUID', cap_col]].copy()

    cap_df[cap_col] = pd.to_numeric(cap_df[cap_col], errors='coerce')

    cap_df = cap_df.dropna(subset=[cap_col])


    # ── Hybrid capacity map ───────────────────────────────────────────────────────

    # AEMO's mid-2024 reclassification of batteries as Bidirectional Units (BDUs)

    # removed all legacy 'G1' generator DUIDs from the NEM Registration and Exemption

    # List entirely. A search across all sheets of the current registry confirms zero

    # matches for these DUIDs. Without a fallback, the pre-5MS battery normalisation

    # denominator would be zero, collapsing the chart to a flat line.

    #

    # Fallback values are sourced from the AEMO NEM Generation Information workbook,

    # cross-checked across the January, May, July, October and November 2021 editions.

    # For three DUIDs absent from all 2021 editions (HBESG1, KESSG1, PIBESG1), the

    # registered capacity of their modern successor DUIDs in the live registry is used

    # as the best available proxy. This is documented in the table below.

    #

    # DUID     | MW     | Station                         | Source

    # ---------|--------|---------------------------------|---------------------------

    # BALBG1   |  30.00 | Ballarat ESS                    | AEMO Gen Info Jan-Nov 2021

    # GANNBG1  |  25.33 | Gannawarra ESS                  | AEMO Gen Info Jan-Nov 2021

    # VBBG1    | 300.00 | Victorian Big Battery           | AEMO Gen Info Oct-Nov 2021

    # BULBG1   |  20.00 | Bulgana BESS                    | AEMO Gen Info 2021 (as BULGANA1)

    # HBESG1   | 200.00 | Hazelwood BESS                  | Absent from 2021 Gen Info;

    #          |        |                                 | capacity from modern successor

    #          |        |                                 | HBESS1 in live registry

    # KESSG1   | 185.00 | Kessell Road BESS               | Absent from 2021 Gen Info;

    #          |        |                                 | capacity from modern successor

    #          |        |                                 | KESSB1 in live registry

    # PIBESG1  |   7.00 | Piber BESS                      | Absent from 2021 Gen Info;

    #          |        |                                 | capacity from modern successor

    #          |        |                                 | PIBESS1 in live registry

    #

    # HBESG1/KESSG1/PIBESG1 do not appear in any 2021 AEMO Generation Information

    # edition, suggesting they were registered or renamed after that period.

    # Using the modern successor registered capacity is a reasonable proxy — it may

    # slightly overstate pre-5MS capacity if the asset was smaller at commissioning,

    # which would conservatively understate the pre-5MS normalised utilisation rate.

    legacy_capacity_fallback = {

        'BALBG1':  30.00,  # Ballarat ESS — AEMO Gen Info Jan-Nov 2021

        'GANNBG1': 25.33,  # Gannawarra ESS — AEMO Gen Info Jan-Nov 2021

        'VBBG1':  300.00,  # Victorian Big Battery — AEMO Gen Info Oct-Nov 2021

        'BULBG1':  20.00,  # Bulgana BESS — AEMO Gen Info 2021

        'HBESG1': 200.00,  # Hazelwood BESS — capacity from modern successor HBESS1

        'KESSG1': 185.00,  # Kessell Road BESS — capacity from modern successor KESSB1

        'PIBESG1':  7.00,  # Piber BESS — capacity from modern successor PIBESS1

    }


    # Build from fallback first, then overwrite with live registry (live takes priority)

    capacity_map = legacy_capacity_fallback.copy()

    capacity_map.update(cap_df.set_index('DUID')[cap_col].to_dict())


    # Report provenance clearly for each DUID

    missing = [d for d in all_relevant_duids if d not in capacity_map]

    if missing:

        missing_bat  = [d for d in missing if d in battery_gen_duids]

        missing_ocgt = [d for d in missing if d in ocgt_duids]

        if missing_bat:

            print(f"\nWARNING: Battery DUIDs with no capacity after fallback: {sorted(missing_bat)}")

            print("  These will contribute 0 MW to the battery normalisation denominator.")

        if missing_ocgt:

            print(f"\nNOTE: OCGT DUIDs with no registry capacity: {sorted(missing_ocgt)}")

            print("  These are likely retired/renamed units and will be excluded by the"

                  " stability filter anyway.")


    legacy_used  = [d for d in all_relevant_duids if d in legacy_capacity_fallback

                    and d not in cap_df.set_index('DUID').index]

    registry_used = [d for d in all_relevant_duids if d in cap_df.set_index('DUID').index]

    print(f"\nCapacity provenance:")

    print(f"  Live registry ({len(registry_used)} DUIDs): {sorted(registry_used)}")

    print(f"  Legacy fallback ({len(legacy_used)} DUIDs): {sorted(legacy_used)}")


    print(f"\nCapacity map ({len(capacity_map)} DUIDs):")

    for duid, mw in sorted(capacity_map.items()):

        print(f"  {duid}: {mw:.1f} MW")


    print(f"\nBattery discharge DUIDs ({len(battery_gen_duids)}): {sorted(battery_gen_duids)}")

    print(f"OCGT DUIDs ({len(ocgt_duids)}):                     {sorted(ocgt_duids)}")
    return battery_gen_duids, capacity_map, ocgt_duids


@app.cell
def _(
    CACHE_DIR,
    POST_END,
    POST_START,
    PRE_END,
    PRE_START,
    battery_gen_duids,
    dynamic_data_compiler,
    mo,
    ocgt_duids,
    pd,
):
    # ── CELL 3: FETCH DISPATCH VOLUMES ────────────────────────────────────────────

    mo.md("## Step 2: Fetch Dispatch Volumes & Apply Stability Filters")


    all_duids = battery_gen_duids + ocgt_duids


    def fetch_dispatch(start, end, duids, label=''):

        print(f"Fetching DISPATCHLOAD [{label}] {start[:10]} → {end[:10]} ...")

        start_dt, end_dt = pd.to_datetime(start), pd.to_datetime(end)

        chunks = []

        current_start = start_dt

        while current_start < end_dt:

            current_end = min(current_start + pd.DateOffset(months=6), end_dt)

            df_chunk = dynamic_data_compiler(

                start_time=current_start.strftime('%Y/%m/%d %H:%M:%S'),

                end_time=current_end.strftime('%Y/%m/%d %H:%M:%S'),

                table_name='DISPATCHLOAD',

                raw_data_location=CACHE_DIR,

                select_columns=['SETTLEMENTDATE', 'DUID', 'TOTALCLEARED'],

                filter_cols=['DUID'],

                filter_values=[duids]

            )

            chunks.append(df_chunk)

            current_start = current_end


        df = pd.concat(chunks, ignore_index=True)

        df['SETTLEMENTDATE'] = pd.to_datetime(df['SETTLEMENTDATE'], errors='coerce')

        df = df.dropna(subset=['SETTLEMENTDATE'])

        df['TOTALCLEARED'] = pd.to_numeric(df['TOTALCLEARED'], errors='coerce').fillna(0)

        return df


    dispatch_pre  = fetch_dispatch(PRE_START,  PRE_END,  all_duids, 'PRE')

    dispatch_post = fetch_dispatch(POST_START, POST_END, all_duids, 'POST')


    # OCGT Stability Filter: active dispatch in ≥2,000 intervals in BOTH periods

    # This prevents units that only ran briefly from skewing the fleet DWP.

    def filter_to_stable_cohort(df_pre, df_post, duids, min_intervals=2000):

        pre_counts  = df_pre[(df_pre['DUID'].isin(duids))  & (df_pre['TOTALCLEARED']  > 0)]['DUID'].value_counts()

        post_counts = df_post[(df_post['DUID'].isin(duids)) & (df_post['TOTALCLEARED'] > 0)]['DUID'].value_counts()

        stable   = set(pre_counts[pre_counts   >= min_intervals].index) & \
                   set(post_counts[post_counts >= min_intervals].index)

        excluded = (set(pre_counts.index) | set(post_counts.index)) - stable

        print(f"  Stable OCGTs:   {sorted(stable)}")

        print(f"  Excluded OCGTs: {sorted(excluded)}")

        return list(stable)


    stable_ocgt = filter_to_stable_cohort(dispatch_pre, dispatch_post, ocgt_duids)


    # Batteries: include any DUID that appears in either period's fetched data

    all_fetched = set(dispatch_pre['DUID'].unique()) | set(dispatch_post['DUID'].unique())

    stable_battery_gen = [d for d in battery_gen_duids if d in all_fetched]


    print(f"\nFinal battery discharge DUIDs ({len(stable_battery_gen)}): {sorted(stable_battery_gen)}")

    print(f"Final stable OCGT DUIDs       ({len(stable_ocgt)}):         {sorted(stable_ocgt)}")
    return dispatch_post, dispatch_pre, stable_battery_gen, stable_ocgt


@app.cell
def _(
    CACHE_DIR,
    POST_END,
    POST_START,
    PRE_END,
    PRE_START,
    VIC_REGION,
    dynamic_data_compiler,
    mo,
    pd,
):
    # ── CELL 4: FETCH SPOT PRICES ─────────────────────────────────────────────────

    mo.md("""

    ## Step 3: Fetch Spot Prices


    - **Pre-5MS:** 30-minute `TRADINGPRICE` — the actual settlement price paid.

    - **Post-5MS:** 5-minute `DISPATCHPRICE` — the new settlement mechanism.

    """)


    def fetch_price(start, end, region, table_name):

        print(f"Fetching {table_name} {start[:10]} → {end[:10]} ...")

        df = dynamic_data_compiler(

            start_time=start, end_time=end,

            table_name=table_name, raw_data_location=CACHE_DIR,

            select_columns=['SETTLEMENTDATE', 'REGIONID', 'RRP'],

            filter_cols=['REGIONID'], filter_values=[[region]]

        )

        df['SETTLEMENTDATE'] = pd.to_datetime(df['SETTLEMENTDATE'])

        df['RRP'] = pd.to_numeric(df['RRP'], errors='coerce')

        return df


    trading_price_pre   = fetch_price(PRE_START,  PRE_END,  VIC_REGION, 'TRADINGPRICE')

    dispatch_price_post = fetch_price(POST_START, POST_END, VIC_REGION, 'DISPATCHPRICE')


    print(f"\nMean VIC RRP pre-5MS:  ${trading_price_pre['RRP'].mean():.2f}/MWh")

    print(f"Mean VIC RRP post-5MS: ${dispatch_price_post['RRP'].mean():.2f}/MWh")
    return dispatch_price_post, trading_price_pre


@app.cell
def _(
    dispatch_post,
    dispatch_pre,
    dispatch_price_post,
    mo,
    stable_battery_gen,
    stable_ocgt,
    trading_price_pre,
):
    # ── CELL 5: MERGE PRICES & CLASSIFY ───────────────────────────────────────────

    mo.md("## Step 4: Merge Prices & Classify Technologies")


    def join_pre(dispatch_df, price_df):

        """

        Map each 5-min dispatch interval to its 30-min trading period price.

        TRADINGPRICE SETTLEMENTDATE is the END of the trading period, which

        aligns correctly with ceil('30min') applied to dispatch timestamps.

        """

        df = dispatch_df.copy()

        df['TRADING_INTERVAL'] = df['SETTLEMENTDATE'].dt.ceil('30min')

        merged = df.merge(

            price_df[['SETTLEMENTDATE', 'RRP']].rename(columns={'SETTLEMENTDATE': 'TRADING_INTERVAL'}),

            on='TRADING_INTERVAL', how='left'

        )

        null_rate = merged['RRP'].isna().mean() * 100

        print(f"Pre-5MS price join: {null_rate:.2f}% of intervals missing price data.")

        if null_rate > 2:

            print("  WARNING: High null rate — verify TRADINGPRICE date alignment.")

        return merged.dropna(subset=['RRP'])


    def join_post(dispatch_df, price_df):

        merged = dispatch_df.merge(

            price_df[['SETTLEMENTDATE', 'RRP']], on='SETTLEMENTDATE', how='left'

        )

        null_rate = merged['RRP'].isna().mean() * 100

        print(f"Post-5MS price join: {null_rate:.2f}% of intervals missing price data.")

        return merged.dropna(subset=['RRP'])


    pre_merged  = join_pre(dispatch_pre,  trading_price_pre)

    post_merged = join_post(dispatch_post, dispatch_price_post)


    def classify_tech(df):

        df = df.copy()

        df['TECH'] = None

        df.loc[df['DUID'].isin(stable_ocgt),       'TECH'] = 'OCGT'

        df.loc[df['DUID'].isin(stable_battery_gen), 'TECH'] = 'Battery'

        return df.dropna(subset=['TECH'])


    pre_merged  = classify_tech(pre_merged)

    post_merged = classify_tech(post_merged)
    return post_merged, pre_merged


@app.cell
def _(
    PERIOD_A_END,
    PERIOD_A_START,
    PERIOD_B_END,
    PERIOD_B_START,
    PERIOD_C_END,
    PERIOD_C_START,
    mo,
    pd,
    post_merged,
    pre_merged,
):
    # ── CELL 6: DWP BY SUB-PERIOD ─────────────────────────────────────────────────

    mo.md("""

    ## Step 5: DWP by Sub-Period


    Three periods are presented to give a complete and transparent picture.

    The crisis-affected period (B) is included but clearly labelled, so the

    reader can see its effect rather than having it silently removed.

    """)


    def calculate_dwp(df, period_label, start, end):

        """

        Compute DWP and a price-tier breakdown for each technology

        within a given date window.

        """

        window = df[

            (df['SETTLEMENTDATE'] >= start) &

            (df['SETTLEMENTDATE'] <  end)   &

            (df['TOTALCLEARED']   >  0)

        ].copy()

        window['MWh'] = window['TOTALCLEARED'] * (5/60)


        rows = []

        for tech in ['Battery', 'OCGT']:

            t = window[window['TECH'] == tech]

            total_mwh = t['MWh'].sum()

            if total_mwh == 0:

                continue


            dwp = (t['MWh'] * t['RRP']).sum() / total_mwh


            # Price-tier breakdown: share of MWh dispatched above each threshold

            pct_above_300  = t.loc[t['RRP'] >  300,  'MWh'].sum() / total_mwh * 100

            pct_above_1000 = t.loc[t['RRP'] > 1000,  'MWh'].sum() / total_mwh * 100

            pct_above_5000 = t.loc[t['RRP'] > 5000,  'MWh'].sum() / total_mwh * 100


            rows.append({

                'Period':                    period_label,

                'Technology':                tech,

                'Total Discharged (MWh)':    round(total_mwh, 1),

                'DWP ($/MWh)':               round(dwp, 2),

                '% MWh > $300':              round(pct_above_300,  1),

                '% MWh > $1,000':            round(pct_above_1000, 1),

                '% MWh > $5,000':            round(pct_above_5000, 1),

            })

        return rows


    all_rows = (

        calculate_dwp(pre_merged,  'A — Pre-5MS (Oct 2019–Oct 2021)',            PERIOD_A_START, PERIOD_A_END) +

        calculate_dwp(post_merged, 'B — Post-5MS Early / Crisis (Oct 2021–Dec 2022)', PERIOD_B_START, PERIOD_B_END) +

        calculate_dwp(post_merged, 'C — Post-5MS Normal (Jan 2023–Oct 2024)',    PERIOD_C_START, PERIOD_C_END)

    )


    summary_df = pd.DataFrame(all_rows)


    print(summary_df.to_string(index=False))

    mo.ui.table(summary_df)
    return


@app.cell
def _(
    PERIOD_B_END,
    PERIOD_B_START,
    capacity_map,
    mo,
    pd,
    plt,
    post_merged,
    pre_merged,
    ticker,
):
    # ── CELL 7: CAPACITY-NORMALISED VOLUME CHART ───────────────────────────────────

    mo.md("""

    ## Step 6: Capacity-Normalised Monthly Discharge Volume


    Raw MWh volume conflates two effects: (1) more batteries being built, and

    (2) existing batteries cycling more frequently due to 5MS. To isolate the

    behavioural effect, volume is normalised by the registered capacity (MW)

    of the active fleet in each month, producing **MWh discharged per MW of

    installed capacity**. A rising trend in this metric reflects more intensive

    utilisation, not just fleet growth.


    Registered capacities use a **hybrid provenance model** to handle AEMO's

    mid-2024 reclassification of batteries as Bidirectional Units (BDUs), which

    removed legacy 'G1' generator DUIDs from the current registry. The map is

    seeded with hardcoded fallback values for those known legacy DUIDs (sourced

    from archived AEMO Generation Information pages), then overwritten with live

    registry figures where available — so the live registry always takes priority.

    The provenance of each DUID's capacity figure is printed in the Step 1 output.

    """)


    # ── 1. capacity_map is passed in from Cell 2 (registry-derived) ──────────────

    # Active capacity = sum of registered MW for DUIDs that dispatched > 0 that month.

    pre_active  = pre_merged[pre_merged['TOTALCLEARED']  > 0].copy()

    post_active = post_merged[post_merged['TOTALCLEARED'] > 0].copy()

    all_active  = pd.concat([pre_active, post_active])

    all_active['Month'] = all_active['SETTLEMENTDATE'].dt.to_period('M').dt.to_timestamp()


    monthly_active_cap = (

        all_active.groupby(['Month', 'TECH'])['DUID']

        .apply(lambda x: sum(capacity_map.get(d, 0) for d in x.unique()))

        .unstack()

        .fillna(0)

    )


    # ── 2. Raw monthly MWh ────────────────────────────────────────────────────────

    monthly_mwh = (

        all_active.groupby(['Month', 'TECH'])

        .apply(lambda x: (x['TOTALCLEARED'] * (5/60)).sum())

        .unstack()

        .fillna(0)

    )


    # ── 3. Normalise: MWh per MW of installed capacity ───────────────────────────

    monthly_norm = (monthly_mwh / monthly_active_cap).replace([float('inf'), float('nan')], 0)


    # ── 4. Plot ───────────────────────────────────────────────────────────────────

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 11), sharex=True)

    rule_change = pd.to_datetime('2021-10-01')


    def shade_crisis(ax):

        """Shade the crisis period instead of excising it."""

        ax.axvspan(PERIOD_B_START, PERIOD_B_END,

                   color='orange', alpha=0.12, label='Crisis period (Oct 2021–Dec 2022)')


    def add_rule_line(ax):

        ax.axvline(rule_change, color='black', linestyle='--',

                   linewidth=1.5, label='5MS Implementation (Oct 2021)')


    # OCGT panel

    ax1.plot(monthly_norm.index, monthly_norm.get('OCGT', 0),

             color='#d9534f', linewidth=2)

    ax1.fill_between(monthly_norm.index, monthly_norm.get('OCGT', 0),

                     color='#d9534f', alpha=0.1)

    shade_crisis(ax1)

    add_rule_line(ax1)

    ax1.set_title("OCGT Fleet: Capacity-Normalised Monthly Discharge\n(MWh discharged per MW of registered capacity)",

                  fontsize=13, pad=10)

    ax1.set_ylabel("MWh / MW", fontsize=11)

    ax1.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.1f}'))

    ax1.legend(loc='upper left', fontsize=9)

    ax1.grid(True, alpha=0.3)


    # Battery panel

    ax2.plot(monthly_norm.index, monthly_norm.get('Battery', 0),

             color='#0275d8', linewidth=2)

    ax2.fill_between(monthly_norm.index, monthly_norm.get('Battery', 0),

                     color='#0275d8', alpha=0.2)

    shade_crisis(ax2)

    add_rule_line(ax2)

    ax2.set_title("Battery Fleet: Capacity-Normalised Monthly Discharge\n(MWh discharged per MW of registered capacity)",

                  fontsize=13, pad=10)

    ax2.set_ylabel("MWh / MW", fontsize=11)

    ax2.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.1f}'))

    ax2.legend(loc='upper left', fontsize=9)

    ax2.grid(True, alpha=0.3)


    ax2.set_xlabel("Month", fontsize=11)

    plt.tight_layout()

    fig
    return


@app.cell
def _(
    PERIOD_A_END,
    PERIOD_A_START,
    PERIOD_C_END,
    PERIOD_C_START,
    mo,
    np,
    pd,
    plt,
    post_merged,
    pre_merged,
    ticker,
):
    # ── CELL 8: PRICE DISTRIBUTION CHART ──────────────────────────────────────────

    mo.md("""

    ## Step 7: Price Distribution — Where Each Fleet Earns Its Volume


    This chart shows what **share of each fleet's discharged MWh** fell within

    each price band, for Pre-5MS (Period A) and Post-5MS Normal (Period C).

    It directly illustrates *why* the DWPs differ: batteries concentrate their

    dispatch at high-price intervals, while OCGTs spread volume across the full

    price range.


    The crisis period (B) is excluded from this comparison to avoid confounding

    the structural 5MS effect with the gas-price shock.

    """)


    # Price bands ($/MWh)

    bins   = [float('-inf'), 0, 100, 300, 1000, 5000, float('inf')]

    labels = ['≤$0 (neg.)', '$0–$100', '$100–$300', '$300–$1k', '$1k–$5k', '>$5k']


    def price_dist(df, start, end, tech):

        window = df[

            (df['SETTLEMENTDATE'] >= start) &

            (df['SETTLEMENTDATE'] <  end)   &

            (df['TOTALCLEARED']   >  0)     &

            (df['TECH']           == tech)

        ].copy()

        window['MWh']  = window['TOTALCLEARED'] * (5/60)

        window['Band'] = pd.cut(window['RRP'], bins=bins, labels=labels)

        dist = window.groupby('Band', observed=True)['MWh'].sum()

        total = dist.sum()

        return (dist / total * 100).reindex(labels).fillna(0) if total > 0 else pd.Series(0, index=labels)


    # Compute distributions for each tech × period combination

    bat_pre  = price_dist(pre_merged,  PERIOD_A_START, PERIOD_A_END, 'Battery')

    ocgt_pre = price_dist(pre_merged,  PERIOD_A_START, PERIOD_A_END, 'OCGT')

    bat_post = price_dist(post_merged, PERIOD_C_START, PERIOD_C_END, 'Battery')

    ocgt_post= price_dist(post_merged, PERIOD_C_START, PERIOD_C_END, 'OCGT')


    x   = np.arange(len(labels))

    w   = 0.2   # bar width

    fig2, ax = plt.subplots(figsize=(13, 6))


    ax.bar(x - 1.5*w, bat_pre,  w, label='Battery — Pre-5MS',         color='#0275d8', alpha=0.5)

    ax.bar(x - 0.5*w, bat_post, w, label='Battery — Post-5MS Normal',  color='#0275d8', alpha=0.95)

    ax.bar(x + 0.5*w, ocgt_pre,  w, label='OCGT — Pre-5MS',           color='#d9534f', alpha=0.5)

    ax.bar(x + 1.5*w, ocgt_post, w, label='OCGT — Post-5MS Normal',   color='#d9534f', alpha=0.95)


    ax.set_xticks(x)

    ax.set_xticklabels(labels, fontsize=10)

    ax.set_xlabel("Spot Price Band ($/MWh)", fontsize=11)

    ax.set_ylabel("Share of Total Discharged MWh (%)", fontsize=11)

    ax.set_title("Price Band Distribution of Discharged MWh\nBattery vs OCGT — Pre-5MS vs Post-5MS Normal",

                 fontsize=13, pad=12)

    ax.legend(fontsize=9)

    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.1f}%'))

    ax.grid(True, axis='y', alpha=0.3)


    plt.tight_layout()

    fig2
    return


if __name__ == "__main__":
    app.run()
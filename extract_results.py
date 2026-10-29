import pandas as pd 
import json
import numpy as np 
import pathlib
import os 
from pathlib import Path
from scipy import stats
LOGS_DIR=Path.home()/"RP3 Rowing Experiment"/"logs"

""" 
    Rhythm: 
        Independent variable: Metronome 
        Dependent variable: Drive time, recovery time, compound metric ratio of drive and recovery time 
        What we want to know: 1. With the feedback is the drive:recovery ratio closer to target than in the baseline 
    
    Force curve:
        Independent variable: ?
        Dependent variable: Root means error, rowing curve of rower compared to ideal form  
        What we want to know: 1. With the feedback is the force curve of rower closer to the 'perfect' force cruve than in the baseline 
        
    Peak force:
        Independent variable: ?  
        Dependent variable: Timing of peak, power of peak  
        What we want to know: 1. With the feedback is the peak force closer to the window than the baseline 
                              2. With the feedback is the power of the peak more consistent than the baseline 
        
    The CSV file contains: 
    "timestampMs", "powerWatts", "paceSecondsPer500Meters", "strokesPerMinute",
    "driveTimeSeconds", "recoverTimeSeconds", "peakForceNewtons", "strokeLengthMeters",
    "peakForcePositionMeters", "relativePeakForcePositionFraction", "energyJoules", "dragFactor",
    "forceCurve", "peakTargetHit", "peakPositionError", "peakPositionErrorPercent",
    "strokeMeanForceNewtons", "baselineMeanForceNewtons", "forceInRange", "peakHitWithForceGate",
    "driveTimeErrorPercent", "driveTimeOnTarget"
  
    Statistical test: 
    In general we want to know if a KPI changes the behaviour of the rower and if possible check if the technique is improved by that behaviour. 

    It is an within subject, because there is a lot of difference between novice rowers technique. 
        Comparing between subjects will have a lot of variation and won't give results we want.  

    Tests:
        1. Each intervention against baseline 
        2. Baseline vs post-baseline to see if fatigue or if changes show learning (might be too short to say learning)
    
    We will remove the first 10 (~30s) strokes from each condition because the rower needs time to adjust. 
    We will check for outliers and remove strokes that are impossible (like >60 spm)

"""

def rhythm(dataframe):
    # Calculate the mean absolute error of drive time (%). Lower is better, 0 is perfect 
    # We calculate the average of how far we are of the perfect drive time for all strokes. We take the abs because -20% and 20% average will give zero so looks perfect but it isn't with abs that doesn;t happen 
    mae_driveTime = dataframe["driveTimeErrorPercent"].abs().mean()

    # Calculate the mean absolute error of recovery:drive ratio. Lower is better, 0 is perfect  
    # We check how far each strokes ratio is from 2 because pefect ratio would be recovery/drive = 2 for 20 spm
    ratio = dataframe["recoverTimeSeconds"] / dataframe["driveTimeSeconds"]
    target_ratio = 2.0 # At 20 spm because 60/20 = 3 with recovery drive ratio 2:1
    mea_ratio = (ratio - target_ratio).abs().mean()

    # Calculate the consistency of the drive 
    drive = dataframe["driveTimeSeconds"]
    consistency_drive_time = drive.std() / drive.mean()

    # Calculate mean of drive time on target. Higher is better, 100 is perfect 
    # We check how many strokes are on target
    target_frac = dataframe["driveTimeOnTarget"].mean()

    return {"mae_driveTime_rhythm_percentage": mae_driveTime, "mea_ratio_rhythm": mea_ratio, "target_frac_rhythm": target_frac, "consistency_drive_time": consistency_drive_time}

def peak_force(dataframe):
    # Get the position of the peak force in fraction. Lower is better, 0 is perfect
    # If the peak is inside the window (34% - 42%) the error is zero. If too early or late then the error is the distance to the window 
    peak_position = dataframe["relativePeakForcePositionFraction"]
    too_early = (0.34 - peak_position).clip(lower=0)
    too_late = (peak_position - 0.42).clip(lower=0)
    mean_error = (too_early + too_late).mean()
    
    # Calculate the consistency of the force peak 
    consistency_peak_timing = dataframe["relativePeakForcePositionFraction"].std()
    
    # The amount of strokes that are inside the window. higher is better, 100 is perfect 
    peak_hit_frac = dataframe["peakTargetHit"].mean()

    # Get the standard deviation of peak force
    std_force = dataframe["peakForceNewtons"].std()
    # get the mean of peak force
    mean_force = dataframe["peakForceNewtons"].mean()

    # Get the consistency of the stroke. Lower is better
    # How much the peak force varies from stroke to stoke divided by the average peak force since that is different for each rower 
    variation = std_force / mean_force

    return {"mean_error_peakForce": mean_error, "peakForce_hit_frac": peak_hit_frac, "variation_peakForce": variation, "consistency_peak_timing": consistency_peak_timing }

# X and Y values of our 'perfect' force curve 
REF_X = [0, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
REF_Y = [5, 35, 60, 75, 86, 97, 100, 96, 87, 74, 57, 35, 20, 5]
XS = np.linspace(0, 100, 201)
 
def smooth(a, w=9):
    pad = w // 2
    return np.convolve(np.pad(a, pad, mode="edge"), np.ones(w) / w, mode="valid")
  
IDEAL = smooth(np.interp(XS, REF_X, REF_Y))



def force_curve(dataframe):
    dataframe_FC = dataframe["forceCurve"]
    rmse, corr = [], []
    curves = []
    for points in dataframe_FC:
        # Since a longer strong has more points we make each curve 100 points so they all run from 0% to 100% of the drive 
        
        x = np.array([p["strokeLengthMeters"] for p in points])
        y = np.array([p["forceNewtons"] for p in points])
        x = (x-x[0]) /(x[-1] - x[0]) * 100 
        y = y / y.max() * 100 
        c = np.interp(XS, x, y)
        curves.append(c)
        # For each point we measure the distance between rowers curve and 'perfect' curve we square it so that all are positive, then take the average and then take the square root 
        # Lower is better because that means it is closer to the 'perfect' force curve
        rmse.append(np.sqrt(np.mean((c - IDEAL) ** 2)))

        # Correlation of force curve, 1 means that is has the same shape -1 means it doesn't have the same shape
        corr.append(np.corrcoef(c, IDEAL)[0,1])

    curves = np.array(curves)                     
    mean_curve = curves.mean(axis=0)     
    # Lower is better means that strokes look more the same       
    consistency = np.mean([np.sqrt(np.mean((c - mean_curve) ** 2)) for c in curves])

    return {"rmse_forceCurve": np.mean(rmse), "correlation_forceCurve": np.mean(corr), "consistency_forceCurve": consistency}


# Read each rower's data and calculate every metric before doing statistics
conditions=[]

for csv_file in LOGS_DIR.glob("*.csv"):
    print(f"Getting data from: {csv_file.name}")

    # Get condition/intervention from filename
    condition=csv_file.stem.split("_",5)[-1]

    dataframe=pd.read_csv(csv_file)

    # Remove first 10 strokes
    dataframe=dataframe.iloc[10:]

    ok=(
        dataframe["driveTimeSeconds"].between(0.3,5.0)
        & dataframe["recoverTimeSeconds"].between(0.3,12.0)
        & dataframe["strokesPerMinute"].between(10,50)
        & (dataframe["peakForceNewtons"]>0)
    )

    dataframe=dataframe[ok].reset_index(drop=True)

    dataframe["forceCurve"]=dataframe["forceCurve"].apply(json.loads)

    # Participant ID from filename
    rower=csv_file.name.split("_")[0]

    row={"rower":rower,"condition":condition}

    row.update(rhythm(dataframe))
    row.update(peak_force(dataframe))
    row.update(force_curve(dataframe))

    row.update({
        "pace_s_per_500m":dataframe["paceSecondsPer500Meters"].mean(),
        "spm":dataframe["strokesPerMinute"].mean(),
        "power_w":dataframe["powerWatts"].mean(),
        "drag_factor":dataframe["dragFactor"].mean(),
        "stroke_length_m":dataframe["strokeLengthMeters"].mean()
    })

    conditions.append(row)

table = pd.DataFrame(conditions)
#table.to_csv("table_per_rower.csv", index=False)


""" 
    Since the rowers are their own control, within subject, we use a paired test
    Since we have about ? rowers we use Wilcoxon signed-rank test 
    If amount of rowers is not an multiple of 6 mention it in the discussion of paper 
        because some orders were used more than others  
    
    Metrics of each KPI:
    Rhythm: mae_driveTime_rhythm, mea_ratio_rhythm, target_frac_rhythm
    Peak force: mean_error_peakForce, peakForce_hit_frac, variation_peakForce
    Force curve: rmse_forceCurve, correlation_forceCurve

    The baseline vs post-baseline is also checked to see if there was fatigue and maybe learning 

    For the metrics target_frac_rhythm, peakForce_hit_frac and correlation_forceCurve a higher 
        is better for all else lower is better
    
    Calculate effect size in the Wilcoxon signed-rank test
    Calculate the other KPI metrics 
    
    Check baseline vs post-baseline 
    Check if force curve intervention imporves consistency force curve of strokes of rower compared to baseline 
    Check if peak force intervention improves consistency compared to baseline 

    TODO: Plot important and useful values/differences 
 
"""


metrics = {
    "Behaviour": {
        "rhythm": ["mae_driveTime_rhythm_percentage"],
        "peak_force": ["mean_error_peakForce"],
        "force_curve": ["rmse_forceCurve"]
    },
    "Consistency": {
        "rhythm": ["consistency_drive_time"],
        "peak_force": ["variation_peakForce", "consistency_peak_timing"],
        "force_curve": ["consistency_forceCurve"]
    },
    "Other KPI metrics": {
        "rhythm": ["mea_ratio_rhythm", "target_frac_rhythm"],
        "peak_force": ["peakForce_hit_frac"],
        "force_curve": ["correlation_forceCurve"]
    },
    "Pace": {
        "rhythm": ["pace_s_per_500m"],
        "peak_force": ["pace_s_per_500m"],
        "force_curve": ["pace_s_per_500m"]
    },
    "Stroke rate": {
        "rhythm": ["spm"],
        "peak_force": ["spm"],
        "force_curve": ["spm"]
    }
}


def effect_size(differences):
    differences = np.asarray(differences, dtype=float)
    differences = differences[differences != 0]

    if len(differences) == 0:
        return np.nan

    ranks = stats.rankdata(np.abs(differences))
    positive = ranks[differences > 0].sum()
    negative = ranks[differences < 0].sum()
    return (positive - negative) / (positive + negative)




def compare(reference, condition, metric, question):
    a = table[table["condition"] == reference].set_index("rower")[metric]
    b = table[table["condition"] == condition].set_index("rower")[metric]
    paired = pd.concat([a.rename("reference"), b.rename("condition")], axis=1).dropna()
    differences = paired["condition"] - paired["reference"]
    n = len(differences)

    statistic, p = np.nan, np.nan
    
    statistic, p = stats.wilcoxon(paired["reference"], paired["condition"])

    
    effect = effect_size(differences)

    n_increased = int((differences > 0).sum())
    n_decreased = int((differences < 0).sum())

    return {
        "question": question,
        "reference": reference,
        "condition": condition,
        "metric": metric,
        "n": n,
        "median_baseline": paired["reference"].median(),
        "median_condition": paired["condition"].median(),
        "n_increased": n_increased,
        "n_decreased": n_decreased,
        "statistic": statistic,
        "p": p,
        "effect_r": effect
    }


#  Dataframe that  contains every comparison
results = []

for question, intervention_metrics in metrics.items():
    for intervention, metric_list in intervention_metrics.items():
        for metric in metric_list:
            results.append(compare("baseline_pre", intervention, metric, question))

# Baseline before vs after, to check for changes over the session
for metric in [
    "mae_driveTime_rhythm_percentage", "mea_ratio_rhythm", "target_frac_rhythm",
    "consistency_drive_time", "mean_error_peakForce", "peakForce_hit_frac",
    "variation_peakForce", "consistency_peak_timing", "rmse_forceCurve",
    "correlation_forceCurve", "consistency_forceCurve", "pace_s_per_500m", "spm"
]:
    results.append(compare("baseline_pre", "baseline_post", metric, "Baseline pre vs post"))

results = pd.DataFrame(results)


# Columns for the result
columns = [
    "condition", "metric", "n", "median_baseline", "median_condition",
    "n_increased", "n_decreased",
    "p", "effect_r"
]


print("\n higher_is_better for target_frac_rhythm, peakForce_hit_frac, correlation_forceCurve, so 1 for effect_r is improved for all others -1 is improvement")
        
for question in results["question"].unique():
    print("---------------------- \n")
    print(f"\n {question}") 
    print(results[results["question"] == question][columns].round(4).to_string(index=False))
    

results.to_csv("results.csv", index=False)

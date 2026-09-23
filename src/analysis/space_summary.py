"""Derived observer and image metrics; never infer target orbits from one line of sight."""
import numpy as np
from astropy.coordinates import EarthLocation
from astropy import units as u


def summarize_space_observation(sequence, run, catalog_fit):
    aux = [sequence.headers[f].space_aux for f in run.frames]
    xyz = np.array([[r['wgs84_'+k] for k in ('x', 'y', 'z')] for r in aux])
    observer = EarthLocation.from_geocentric(*xyz.T, unit=u.m)
    height = observer.height.to_value(u.km)
    speed = np.linalg.norm([[r['j2000_'+k] for k in ('xv', 'yv', 'zv')] for r in aux], axis=1)/1000
    timestamps = np.array([sequence.headers[f].date_obs.unix for f in run.frames])
    # Drift at the actual image center (a translation alone omits rotation).
    shape = sequence.image(run.reference_frame).shape
    center = np.array([[(shape[1]-1)/2, (shape[0]-1)/2]])
    drift = np.array([run.registration.to_sky(f, center)[0]-center[0] for f in run.frames])
    return dict(duration_s=float(timestamps[-1]-timestamps[0]),
                cadence_median_s=float(np.median(np.diff(timestamps))),
                observer_height_above_wgs84_km=[float(height.min()), float(height.max())],
                observer_j2000_speed_km_s=[float(speed.min()), float(speed.max())],
                field_center_drift_span_px=float(np.linalg.norm(np.ptp(drift, axis=0))),
                approximate_scale_arcsec_px=catalog_fit['approximate_scale_arcsec_px'] if catalog_fit else None,
                target_approximate_angular_speeds_arcsec_s=[
                    float(np.linalg.norm(t.velocity)*catalog_fit['approximate_scale_arcsec_px']) for t in run.targets
                ] if catalog_fit else None,
                note='高度和轨道速度属于观测卫星；目标角速度以局部平均像元尺度估计，不给出目标距离、轨道或身份。')

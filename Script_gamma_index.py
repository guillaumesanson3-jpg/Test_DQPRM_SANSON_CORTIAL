"""
╔══════════════════════════════════════════════════════════════════╗
║   GAMMA 3D/2D - Comparaison dosimétrique                         ║
║   Outils : pydicom · scipy · matplotlib · numpy                  ║
║   Réf : Low et al. Med Phys 1998, AAPM TG-218, ICRU 83,          ║
║          AAPM TG-132, Robar et al. 2024                          ║
╚══════════════════════════════════════════════════════════════════╝

Prérequis : pip install pydicom matplotlib numpy scipy

CORRECTIONS appliquées :
  - RTSTRUCT : correction du référentiel isocentre pour les contours
  - RTSTRUCT : filtrage des ROI exclues (Isocentre, Contour_externe...)
  - Gamma : mode 2D / 3D / both au choix
  - Métriques : calculées uniquement sur les vraies structures contourées
  - DVH : une courbe par structure avec couleur dédiée
  
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.path import Path
import pydicom
import warnings
import os
from scipy.interpolate import RegularGridInterpolator



warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — À modifier selon vos fichiers
# ─────────────────────────────────────────────────────────────────────────────

CONFIG = {
    # ── Fichiers DICOM ────────────────────────────────────────────────────────
    "dose_reference":    "G:/BUREAU/DOSI/Stage M2 2026/Etude dosimétrique/Fichiers_export/Cheese/Plans PTVs/images CT/D_courbe_2018/DIAMCHEESE2026_ScanPTVD2018Courbe_Dose.dcm",
    "dose_evaluation":   "G:/BUREAU/DOSI/Stage M2 2026/Etude dosimétrique/Fichiers_export/Cheese/Plans PTVs/Image CBCT/D_courbe_Iris/DIAMCHEESE2026_XVIPTVDIrisCourbe_Dose.dcm",
    "rtplan_reference":  "G:/BUREAU/DOSI/Stage M2 2026/Etude dosimétrique/Fichiers_export/Cheese/Plans PTVs/images CT/D_courbe_2018/DIAMCHEESE2026_ScanPTVD2018Courbe.dcm",
    "rtplan_evaluation": "G:/BUREAU/DOSI/Stage M2 2026/Etude dosimétrique/Fichiers_export/Cheese/Plans PTVs/Image CBCT/D_courbe_Iris/DIAMCHEESE2026_XVIPTVDIrisCourbe.dcm",
    "rtss_file":         "G:/BUREAU/DOSI/Stage M2 2026/Etude dosimétrique/Fichiers_export/Cheese/Plans PTVs/DIAMCHEESE2026_StrctrSets.dcm",

    # ── Labels graphiques ─────────────────────────────────────────────────────
  
    "label_ref":  "CT Simulation",
    "label_eval": "CBCT XVI Iris",
    "plan_name":  "ROI Fond_17 GAUCHE Cheese - Dosi CT Vs Iris",

    # ── Gamma ─────────────────────────────────────────────────────────────────
    # Mode : "3D"   : volumique (AAPM TG-218, recommandé pour dosimétrie patient)
    #        "2D"   : plan par plan axial (style Doselab / SNC Patient)
    #        "both" : les deux 
    "gamma_mode":             "3D",
    "dose_threshold_percent": 3.0,    # ΔD (%) — critère global
    "distance_mm":            3.0,    # Δd (mm)
    "lower_dose_cutoff":      10.0,   # seuil bas (% Dmax) — AAPM TG-218
    "max_gamma":              10,    # plafond affiché
    "n_shells":               20,     # précision shells 3D 
    "n_angles":               50,     # directions par shell (Fibonacci)

    # ── Dose de prescription ──────────────────────────────────────────────────
    # None = auto (D98 du CT sur le volume global)
    "prescribed_dose_gy": 36,

    # ── Structures à EXCLURE des métriques ───────────────────────────────────
    # Points, isocentres, structures trop grandes ou hors fantôme
    "roi_exclude": ["Isocentre", "Contour_externe"],

    # ── Coupes affichées (None = centrale automatique) ────────────────────────
    "slice_axial":    None,
    "slice_sagittal": None,
    "slice_coronal":  None,

    # ── Sortie ────────────────────────────────────────────────────────────────

    "output_dir": "G:/BUREAU/DOSI/Stage M2 2026/Etude dosimétrique/resultats/test",
    "dpi": 150,
}


# ─────────────────────────────────────────────────────────────────────────────
# COLORMAPS
# ─────────────────────────────────────────────────────────────────────────────

def make_dose_cmap():
    colors = [(0.0,(0,0,.5)),(0.1,(0,0,1)),(0.3,(0,.8,.8)),
              (0.5,(0,.9,0)),(0.7,(1,1,0)),(0.85,(1,.5,0)),(1.0,(1,0,0))]
    return LinearSegmentedColormap('dose_tps', {
        'red':  [(p,c[0],c[0]) for p,c in colors],
        'green':[(p,c[1],c[1]) for p,c in colors],
        'blue': [(p,c[2],c[2]) for p,c in colors]})

def make_gamma_cmap():
    colors = [(0.0,(.05,.7,.05)),(0.5,(1,1,0)),(0.9,(1,.4,0)),(1.0,(.9,0,0))]
    return LinearSegmentedColormap('gamma_map', {
        'red':  [(p,c[0],c[0]) for p,c in colors],
        'green':[(p,c[1],c[1]) for p,c in colors],
        'blue': [(p,c[2],c[2]) for p,c in colors]})

CMAP_DOSE  = make_dose_cmap()
CMAP_GAMMA = make_gamma_cmap()

PALETTE = ["#3498db","#e74c3c","#2ecc71","#f39c12","#9b59b6",
           "#1abc9c","#e67e22","#e91e63","#00bcd4","#8bc34a",
           "#ff5722","#607d8b","#795548","#cddc39","#03a9f4"]

# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT DICOM
# ─────────────────────────────────────────────────────────────────────────────

def load_rtdose(filepath):
    print(f"  → Chargement : {os.path.basename(filepath)}")
    from pydicom.uid import ImplicitVRLittleEndian
    ds = pydicom.dcmread(filepath, force=True)
    if not hasattr(ds.file_meta, "TransferSyntaxUID"):
        ds.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian

    modality = getattr(ds, "Modality", "").strip().upper()
    if modality not in ("RTDOSE", ""):
        raise ValueError(f"Pas un RT Dose (Modality='{modality}'): {filepath}")

    scaling = float(getattr(ds, "DoseGridScaling", 1.0))
    dose    = ds.pixel_array.astype(np.float64) * scaling
    origin  = [float(v) for v in ds.ImagePositionPatient]
    spacing = [float(v) for v in ds.PixelSpacing]
    offsets = [float(v) for v in ds.GridFrameOffsetVector]

    x_axis = origin[0] + np.arange(ds.Columns) * spacing[1]
    y_axis = origin[1] + np.arange(ds.Rows)    * spacing[0]
    z_axis = origin[2] + np.array(offsets)

    print(f"     Grille : {dose.shape} | Dmax : {dose.max():.3f} Gy | "
          f"{spacing[0]:.1f}x{spacing[1]:.1f} mm | {len(offsets)} coupes "
          f"| Z=[{z_axis[0]:.1f}…{z_axis[-1]:.1f}]")
    return (z_axis, y_axis, x_axis), dose, ds


def get_isocenter(rtplan_path):
    if rtplan_path is None:
        return None
    try:
        ds = pydicom.dcmread(rtplan_path, force=True)
        iso = ds.BeamSequence[0].ControlPointSequence[0].IsocenterPosition
        coords = np.array([float(v) for v in iso])
        print(f"     Isocentre : ({coords[0]:.2f}, {coords[1]:.2f}, {coords[2]:.2f}) mm")
        return coords  # (x, y, z) DICOM
    except Exception as e:
        print(f"  ⚠  Isocentre illisible ({e})")
        return None


def resample_eval_on_ref(axes_ref, axes_eval, dose_eval, iso_ref=None, iso_eval=None):
    """Resamplise dose_eval sur axes_ref.

    Stratégie de correction du décalage (par ordre de priorité) :
    1. Si les deux isocentres sont fournis → décalage exact via isocentres
    2. Sinon → décalage estimé depuis la différence d'origines DICOM
       (centre de la grille de chaque dose)
    Couvre tous les cas : même spacing, spacings différents, origines décalées.
    """
    z_ev, y_ev, x_ev = [np.array(a, dtype=float) for a in axes_eval]
    z_ref, y_ref, x_ref = [np.array(a, dtype=float) for a in axes_ref]

    # Assurer axes croissants
    if z_ev[-1] < z_ev[0]: z_ev = z_ev[::-1]; dose_eval = dose_eval[::-1, :, :]
    if y_ev[-1] < y_ev[0]: y_ev = y_ev[::-1]; dose_eval = dose_eval[:, ::-1, :]
    if x_ev[-1] < x_ev[0]: x_ev = x_ev[::-1]; dose_eval = dose_eval[:, :, ::-1]

    if iso_ref is not None and iso_eval is not None:
        # Méthode 1 — isocentres connus : décalage exact
        dx = iso_ref[0] - iso_eval[0]   # x=LR
        dy = iso_ref[1] - iso_eval[1]   # y=AP
        dz = iso_ref[2] - iso_eval[2]   # z=SI
        print(f"  → Décalage via isocentres : ΔX={dx:.2f} ΔY={dy:.2f} ΔZ={dz:.2f} mm")
    else:
        # Méthode 2 — pas d'isocentre : on suppose que les centres des grilles
        # correspondent (hypothèse raisonnable si même fantôme, même positionnement)
        dx = (x_ref[0]+x_ref[-1])/2 - (x_ev[0]+x_ev[-1])/2
        dy = (y_ref[0]+y_ref[-1])/2 - (y_ev[0]+y_ev[-1])/2
        dz = (z_ref[0]+z_ref[-1])/2 - (z_ev[0]+z_ev[-1])/2
        print(f"  ⚠  Isocentres absents — décalage estimé par centres de grille :")
        print(f"     ΔX={dx:.2f} ΔY={dy:.2f} ΔZ={dz:.2f} mm")
        print(f"     (vérifier visuellement que l'alignement est correct)")

    interp = RegularGridInterpolator(
        (z_ev, y_ev, x_ev), dose_eval,
        method="linear", bounds_error=False, fill_value=0.0)

    ZZ, YY, XX = np.meshgrid(z_ref, y_ref, x_ref, indexing="ij")
    pts = np.column_stack([ZZ.ravel()-dz, YY.ravel()-dy, XX.ravel()-dx])
    dose_r = interp(pts).reshape(ZZ.shape)
    print(f"     Resamplé : {dose_r.shape} | Dmax : {dose_r.max():.3f} Gy")
    return dose_r

# ─────────────────────────────────────────────────────────────────────────────
# LECTURE RTSTRUCT avec correction isocentre
# CORRECTION CLÉ : les contours sont dans le référentiel CBCT (z ~ -151 à +161)
# mais la grille de dose CT est à z ~ +616 à +922. On applique le même décalage
# isocentre que pour la dose afin de ramener les contours dans le bon référentiel.
# ─────────────────────────────────────────────────────────────────────────────

def _read_rtstruct_robust(filepath):
    """Lit un fichier RTSTRUCT avec plusieurs stratégies de décodage.

    Gère les cas :
    - DICOM standard (avec preamble 128B + DICM)
    - DICOM sans preamble (Monaco/Elekta) en ImplicitVR ou ExplicitVR
    - Fichier commençant directement par les données (offset 0)
    """
    import io
    from pydicom.uid import ImplicitVRLittleEndian, ExplicitVRLittleEndian
    from pydicom.filereader import read_dataset
    from pydicom.filebase import DicomBytesIO

    def _has_structs(ds):
        return (hasattr(ds, "StructureSetROISequence") and
                hasattr(ds, "ROIContourSequence") and
                len(ds.StructureSetROISequence) > 0)

    with open(filepath, "rb") as f:
        raw = f.read()

    # ── Stratégie 1 : lecture standard pydicom ────────────────────────────────
    try:
        ds = pydicom.dcmread(filepath, force=False)
        if _has_structs(ds): return ds, "standard"
    except Exception:
        pass

    # ── Stratégie 2 : force=True ──────────────────────────────────────────────
    try:
        ds = pydicom.dcmread(filepath, force=True)
        if _has_structs(ds): return ds, "force=True"
    except Exception:
        pass

    # ── Stratégies 3-10 : lecture bas niveau VR × endian × offset ────────────
    for is_implicit in [True, False]:
        for is_little in [True, False]:
            for offset in [0, 132]:
                try:
                    buf = DicomBytesIO(raw[offset:])
                    buf.is_implicit_VR   = is_implicit
                    buf.is_little_endian = is_little
                    ds = read_dataset(buf, is_implicit, is_little)
                    if _has_structs(ds):
                        tag_name = ("Implicit" if is_implicit else "Explicit")
                        end_name = ("LE" if is_little else "BE")
                        return ds, f"{tag_name}{end_name}+offset{offset}"
                except Exception:
                    pass

    # ── Stratégie 11 : parser brut — chercher le tag (3006,0020) dans les octets
    # Monaco "copie" réexporte parfois avec un header propriétaire de taille
    # variable. On cherche directement la séquence d'octets du tag RTSTRUCT.
    # Tag (3006,0020) en Little Endian = 06 30 20 00
    TAG_SSROI = b"\x06\x30\x20\x00"
    TAG_RCROI = b"\x06\x30\x39\x00"
    pos_ssroi = raw.find(TAG_SSROI)
    pos_rcroi = raw.find(TAG_RCROI)

    if pos_ssroi > 0 and pos_rcroi > 0:
        # Trouver l'offset de début du dataset : chercher le premier tag valide
        # avant StructureSetROISequence. On remonte jusqu'à un tag (0008,xxxx).
        TAG_0008 = b"\x08\x00"
        start_offset = raw.rfind(TAG_0008, 0, pos_ssroi)
        if start_offset < 0:
            start_offset = 0

        for is_implicit in [True, False]:
            try:
                buf = DicomBytesIO(raw[start_offset:])
                buf.is_implicit_VR   = is_implicit
                buf.is_little_endian = True
                ds = read_dataset(buf, is_implicit, True)
                if _has_structs(ds):
                    return ds, f"raw_scan_offset{start_offset}"
            except Exception:
                pass

    return None, "échec_toutes_stratégies"


def load_rtstruct_masks(rtss_path, axes_ref, iso_ref=None, iso_ss=None, exclude=None):
    """Charge les masques ROI depuis un RT Structure Set.

    Applique la correction d'isocentre pour ramener les contours (référentiel CBCT)
    dans le référentiel CT avant rasterisation sur la grille de dose.

    Paramètres
    ----------
    rtss_path : str | None
    axes_ref  : (z_ref, y_ref, x_ref) — grille de dose de référence (CT)
    iso_ref   : isocentre du plan CT  (x, y, z) en mm
    iso_ss    : isocentre du plan CBCT (x, y, z) en mm — référentiel des contours
    exclude   : liste de noms de ROI à ignorer
    """
    if rtss_path is None:
        return {}

    exclude = set(e for e in (exclude or []) if e)  # filtre les chaînes vides
    print(f"  → Lecture RTSTRUCT : {os.path.basename(rtss_path)}")

    # ── Lecture robuste : plusieurs stratégies de décodage ────────────────────
    # Certains exports Monaco/Elekta ont un header DICOM non standard ou un
    # Transfer Syntax privé. On essaie plusieurs modes jusqu'à trouver les ROIs.
    ds = None
    strategies = [
        # (description, kwargs)
        ("standard",          dict(force=False)),
        ("force=True",        dict(force=True)),
        ("force+ImplicitLE",  dict(force=True)),  # Transfer Syntax fixé après
        ("force+ExplicitLE",  dict(force=True)),  # Transfer Syntax fixé après
    ]
    ds, strat_used = _read_rtstruct_robust(rtss_path)
    if ds is not None:
        print(f"     Lecture OK (stratégie : {strat_used})")

    # ── Diagnostic si toujours invalide ──────────────────────────────────────
    if ds is None or not hasattr(ds, "StructureSetROISequence"):
        has_rois     = ds is not None and hasattr(ds, "StructureSetROISequence")
        has_contours = ds is not None and hasattr(ds, "ROIContourSequence")
        modality     = getattr(ds, "Modality", "?") if ds else "?"
        print(f"  ✗ RTSTRUCT illisible — diagnostic :")
        print(f"     Modality             : '{modality}'")
        print(f"     StructureSetROISeq   : {'OK' if has_rois else 'MANQUANT'}")
        print(f"     ROIContourSeq        : {'OK' if has_contours else 'MANQUANT'}")
        if ds is not None:
            print(f"     Tags présents ({len(list(ds))}) :")
            for elem in list(ds)[:30]:
                try:
                    val_str = str(elem.value)[:80]
                except Exception:
                    val_str = "<illisible>"
                print(f"       {elem.tag}  {elem.keyword or '?':32s} = {val_str}")
        print("  → Le script continue sans structures (métriques sur volume global)")
        return {}

    modality = getattr(ds, "Modality", "?").strip().upper()
    print(f"     Modality : '{modality}' | "
          f"{len(ds.StructureSetROISequence)} ROIs | "
          f"{len(ds.ROIContourSequence)} contours")
    # ──────────────────────────────────────────────────────────────────────────

    z_ref, y_ref, x_ref = axes_ref
    nz, ny, nx = len(z_ref), len(y_ref), len(x_ref)

    # ── Calcul du décalage à appliquer aux contours ──────────────────────────
    # Stratégie 1 : via isocentres (si disponibles et différents)
    # Stratégie 2 : via la différence d'origine Z entre le Frame of Reference
    #               du RTSTRUCT et la grille de dose de référence.
    #
    # Pourquoi la stratégie 2 est nécessaire :
    # Quand on utilise un RTSTRUCT dessiné sur le CT pour analyser une dose CBCT,
    # les coordonnées Z des contours sont dans le référentiel CT (ex: +616 à +922mm)
    # mais la grille de dose CBCT est à Z=[-156 à +158mm]. La différence d'origine
    # Z des images CT vs CBCT doit être corrigée, indépendamment de l'isocentre.

    dx = dy = dz = 0.0

    if iso_ref is not None and iso_ss is not None:
        iso_diff = np.sqrt((iso_ref[0]-iso_ss[0])**2 +
                           (iso_ref[1]-iso_ss[1])**2 +
                           (iso_ref[2]-iso_ss[2])**2)
        if iso_diff > 0.5:  # décalage réel entre les deux isocentres
            dx = iso_ref[0] - iso_ss[0]
            dy = iso_ref[1] - iso_ss[1]
            dz = iso_ref[2] - iso_ss[2]
            print(f"     Correction via isocentres : ΔX={dx:.1f} ΔY={dy:.1f} ΔZ={dz:.1f} mm")
        else:
            print(f"     Isocentres identiques (diff={iso_diff:.2f}mm) — pas de correction isocentre")

    # Vérification : est-ce que les contours tombent dans la grille de dose ?
    # On prend le premier contour disponible et on teste sa position Z.
    z_contour_sample = None
    for rc in ds.ROIContourSequence:
        if hasattr(rc, "ContourSequence") and len(rc.ContourSequence) > 0:
            try:
                pts = np.array(rc.ContourSequence[0].ContourData, dtype=float).reshape(-1, 3)
                z_contour_sample = pts[0, 2] + dz
                break
            except Exception:
                pass

    if z_contour_sample is not None:
        z_ref_min, z_ref_max = z_ref.min(), z_ref.max()
        in_grid = z_ref_min - 10 <= z_contour_sample <= z_ref_max + 10

        if not in_grid:
            # Les contours sont hors grille même après correction isocentre.
            # On calcule le décalage Z nécessaire pour les ramener dans la grille.
            # Le centre des contours doit correspondre au centre de la grille de dose.
            z_contour_raw = z_contour_sample - dz  # position brute sans correction

            # Collecter toutes les positions Z des contours pour trouver leur centre
            all_z_contours = []
            for rc2 in ds.ROIContourSequence:
                if hasattr(rc2, "ContourSequence"):
                    for c in rc2.ContourSequence:
                        try:
                            z_c = float(np.array(c.ContourData, dtype=float).reshape(-1,3)[0,2])
                            all_z_contours.append(z_c)
                        except Exception:
                            pass

            if all_z_contours:
                z_center_contours = (min(all_z_contours) + max(all_z_contours)) / 2
                z_center_grid     = (z_ref_min + z_ref_max) / 2
                dz_auto = z_center_grid - z_center_contours

                # Pareil pour X et Y si décalage isocentre nul
                if abs(dx) < 0.5 and abs(dy) < 0.5:
                    all_x = []; all_y = []
                    for rc2 in ds.ROIContourSequence:
                        if hasattr(rc2, "ContourSequence"):
                            for c in rc2.ContourSequence:
                                try:
                                    pts2 = np.array(c.ContourData, dtype=float).reshape(-1,3)
                                    all_x.extend(pts2[:,0].tolist())
                                    all_y.extend(pts2[:,1].tolist())
                                except Exception:
                                    pass
                    if all_x:
                        x_center_c = (min(all_x)+max(all_x))/2
                        y_center_c = (min(all_y)+max(all_y))/2
                        x_center_g = (x_ref.min()+x_ref.max())/2
                        y_center_g = (y_ref.min()+y_ref.max())/2
                        dx_auto = x_center_g - x_center_c
                        dy_auto = y_center_g - y_center_c
                        # N'appliquer X/Y que si décalage significatif (>5mm)
                        if abs(dx_auto) > 5: dx += dx_auto
                        if abs(dy_auto) > 5: dy += dy_auto

                dz += dz_auto
                print(f"     ⚠  Contours hors grille détectés — correction automatique par centres :")
                print(f"        Z contours : [{min(all_z_contours):.1f} … {max(all_z_contours):.1f}] mm")
                print(f"        Z grille   : [{z_ref_min:.1f} … {z_ref_max:.1f}] mm")
                print(f"        ΔZ ajouté  : {dz_auto:.1f} mm")
                print(f"     Correction finale : ΔX={dx:.1f} ΔY={dy:.1f} ΔZ={dz:.1f} mm")
        else:
            print(f"     ✓ Contours dans la grille (Z échantillon après correction : {z_contour_sample:.1f} mm)")

    roi_name_map = {r.ROINumber: r.ROIName for r in ds.StructureSetROISequence}
    masks = {}

    for rc in ds.ROIContourSequence:
        roi_name = roi_name_map.get(rc.ReferencedROINumber, f"ROI_{rc.ReferencedROINumber}")

        if roi_name in exclude:
            print(f"     ↷ '{roi_name}' exclue (liste roi_exclude)")
            continue

        if not hasattr(rc, "ContourSequence") or len(rc.ContourSequence) == 0:
            print(f"     ✗ '{roi_name}' : pas de contours")
            continue

        mask  = np.zeros((nz, ny, nx), dtype=bool)
        n_ok  = 0
        n_skip = 0

        for contour in rc.ContourSequence:
            try:
                pts = np.array(contour.ContourData, dtype=float).reshape(-1, 3)
            except Exception:
                n_skip += 1
                continue

            if len(pts) < 3:
                n_skip += 1
                continue

            # Appliquer la correction isocentre (CBCT → CT)
            pts_corr = pts.copy()
            pts_corr[:, 0] += dx   # x
            pts_corr[:, 1] += dy   # y
            pts_corr[:, 2] += dz   # z

            z_val = pts_corr[0, 2]

            # Vérifier que z est dans la grille de dose (tolérance 1 voxel)
            dz_grid = abs(z_ref[1] - z_ref[0]) if len(z_ref) > 1 else 3.0
            if z_val < z_ref.min() - dz_grid or z_val > z_ref.max() + dz_grid:
                n_skip += 1
                continue

            iz = int(np.argmin(np.abs(z_ref - z_val)))

            # Rasterisation 2D dans le plan (x, y)
            poly_xy = pts_corr[:, :2]
            path_2d  = Path(poly_xy)
            XX2, YY2 = np.meshgrid(x_ref, y_ref)
            inside   = path_2d.contains_points(
                np.column_stack([XX2.ravel(), YY2.ravel()])
            ).reshape(ny, nx)
            mask[iz] |= inside
            n_ok += 1

        if mask.any():
            masks[roi_name] = mask
            print(f"     ✓ '{roi_name}' : {mask.sum():,} voxels ({n_ok} coupes OK, {n_skip} hors grille)")
        else:
            print(f"     ✗ '{roi_name}' : hors grille ou vide après correction ({n_ok} OK, {n_skip} hors grille)")

    print(f"  → {len(masks)} structure(s) chargée(s)")
    return masks

# ─────────────────────────────────────────────────────────────────────────────
# GAMMA 3D — algorithme shells sphériques (Low et al. Med Phys 1998)
# Normalisation globale (Dmax ref) — AAPM TG-218 recommandation pour plans patient
# ─────────────────────────────────────────────────────────────────────────────

def _fibonacci_sphere(n):
    """n directions uniformes sur la sphère unité (spirale de Fibonacci)."""
    golden = np.pi * (3 - np.sqrt(5))
    idx    = np.arange(n)
    y_d    = 1 - 2 * idx / (n - 1)
    r      = np.sqrt(np.maximum(1 - y_d**2, 0))
    th     = golden * idx
    return np.column_stack([r*np.cos(th), y_d, r*np.sin(th)])  # (N, 3) z,y,x


def compute_gamma_3d(axes_ref, dose_ref, axes_eval, dose_eval, cfg):
    """Gamma 3D volumique par shells sphériques.

    Réf : Low et al. Med Phys 1998 / AAPM TG-218
    Normalisation : globale (Dmax ref)
    Early exit : uniquement après avoir exploré jusqu'à dist = DTA
    (évite l'artefact γ=1 exact qui surviendrait avec exit trop précoce)
    """
    dd     = cfg["dose_threshold_percent"] / 100.0
    dta    = cfg["distance_mm"]
    cutoff = cfg["lower_dose_cutoff"] / 100.0
    max_g  = cfg["max_gamma"]
    n_sh   = cfg["n_shells"]
    n_ang  = cfg["n_angles"]

    z_ev, y_ev, x_ev = axes_eval
    interp = RegularGridInterpolator(
        (z_ev, y_ev, x_ev), dose_eval,
        method="linear", bounds_error=False, fill_value=np.nan)

    dose_norm = dose_ref.max()
    dd_abs    = dd * dose_norm                    # seuil absolu en Gy
    mask      = dose_ref >= cutoff * dose_norm    # voxels évalués

    z_ref, y_ref, x_ref = axes_ref
    ZZ, YY, XX = np.meshgrid(z_ref, y_ref, x_ref, indexing="ij")
    gamma_vol  = np.full(dose_ref.shape, np.nan, dtype=np.float32)

    distances  = np.linspace(0, dta * max_g, n_sh + 1)
    directions = _fibonacci_sphere(n_ang)   # (N, 3) — z,y,x

    idx_flat = np.argwhere(mask)
    total    = len(idx_flat)
    print(f"     {total:,} voxels à évaluer (seuil {cfg['lower_dose_cutoff']}% Dmax)...")

    for vi, (iz, iy, ix) in enumerate(idx_flat):
        if vi % max(1, total // 20) == 0:
            print(f"     Progression : {100*vi//total:3d}%", end="\r")

        ref_dose  = dose_ref[iz, iy, ix]
        rc        = np.array([ZZ[iz,iy,ix], YY[iz,iy,ix], XX[iz,iy,ix]])
        best_g2   = np.inf

        for dist in distances:
            if dist == 0.0:
                ev = interp([rc])[0]
                if not np.isnan(ev):
                    g2 = ((ev - ref_dose) / dd_abs) ** 2
                    if g2 < best_g2:
                        best_g2 = g2
            else:
                coords   = rc + dist * directions        # (N, 3)
                ev_doses = interp(coords)                # (N,)
                valid    = ~np.isnan(ev_doses)
                if np.any(valid):
                    g2v = ((ev_doses[valid] - ref_dose) / dd_abs)**2 + (dist/dta)**2
                    gm  = g2v.min()
                    if gm < best_g2:
                        best_g2 = gm

            # Early exit seulement après avoir couvert au moins 1 DTA
            if best_g2 <= 1.0 and dist >= dta:
                break

        gamma_vol[iz, iy, ix] = min(np.sqrt(best_g2), max_g) if best_g2 < np.inf else max_g

    print("     Progression : 100%")
    return gamma_vol


def compute_gamma_2d(axes_ref, dose_ref, axes_eval, dose_eval, cfg):
    """Gamma 2D plan par plan axial — style Doselab / SNC Patient.

    Retourne un volume gamma identique en shape à dose_ref,
    calculé coupe par coupe dans le plan (y, x), z fixé.
    Réf : Low et al. 1998 appliqué en 2D.
    """
    dd     = cfg["dose_threshold_percent"] / 100.0
    dta    = cfg["distance_mm"]
    cutoff = cfg["lower_dose_cutoff"] / 100.0
    max_g  = cfg["max_gamma"]
    n_sh   = cfg["n_shells"]
    n_ang  = cfg["n_angles"]

    dose_norm = dose_ref.max()
    dd_abs    = dd * dose_norm

    z_ref, y_ref, x_ref = axes_ref
    z_ev,  y_ev,  x_ev  = axes_eval
    gamma_vol = np.full(dose_ref.shape, np.nan, dtype=np.float32)

    # Angles 2D uniformes dans le plan
    angles_2d = np.linspace(0, 2*np.pi, n_ang, endpoint=False)
    dirs_2d   = np.column_stack([np.cos(angles_2d), np.sin(angles_2d)])  # (N, 2) y,x
    distances  = np.linspace(0, dta * max_g, n_sh + 1)

    total_slices = dose_ref.shape[0]
    for iz in range(total_slices):
        if iz % max(1, total_slices // 10) == 0:
            print(f"     Coupe {iz}/{total_slices}", end="\r")

        sl_ref  = dose_ref[iz, :, :]
        iz_ev   = int(np.argmin(np.abs(z_ev - z_ref[iz])))
        sl_eval = dose_eval[iz_ev, :, :]

        interp_2d = RegularGridInterpolator(
            (y_ev, x_ev), sl_eval,
            method="linear", bounds_error=False, fill_value=np.nan)

        mask_sl = sl_ref >= cutoff * dose_norm
        iy_arr, ix_arr = np.where(mask_sl)

        for iy, ix in zip(iy_arr, ix_arr):
            ref_dose = sl_ref[iy, ix]
            rc2d     = np.array([y_ref[iy], x_ref[ix]])
            best_g2  = np.inf

            for dist in distances:
                if dist == 0.0:
                    ev = interp_2d([rc2d])[0]
                    if not np.isnan(ev):
                        g2 = ((ev - ref_dose) / dd_abs)**2
                        if g2 < best_g2:
                            best_g2 = g2
                else:
                    coords = rc2d + dist * dirs_2d   # (N, 2)
                    evs    = interp_2d(coords)        # (N,)
                    valid  = ~np.isnan(evs)
                    if np.any(valid):
                        g2v = ((evs[valid] - ref_dose) / dd_abs)**2 + (dist/dta)**2
                        gm  = g2v.min()
                        if gm < best_g2:
                            best_g2 = gm

                if best_g2 <= 1.0 and dist >= dta:
                    break

            gamma_vol[iz, iy, ix] = min(np.sqrt(best_g2), max_g) if best_g2 < np.inf else max_g

    print(f"     Coupe {total_slices}/{total_slices} — terminé")
    return gamma_vol


def gamma_stats(gamma_vol):
    valid = gamma_vol[~np.isnan(gamma_vol)]
    if len(valid) == 0:
        return {"n_voxels_evaluated":0,"pass_rate":0.0,"mean":0.0,"median":0.0,"max":0.0,"p95":0.0}
    return {
        "n_voxels_evaluated": len(valid),
        "pass_rate": 100.0 * np.sum(valid <= 1.0) / len(valid),
        "mean":   float(np.mean(valid)),
        "median": float(np.median(valid)),
        "max":    float(np.max(valid)),
        "p95":    float(np.percentile(valid, 95)),
    }

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 : Cartes de dose + gamma
# ─────────────────────────────────────────────────────────────────────────────

def _sl(vol, axis, override):
    return override if override is not None else vol.shape[axis] // 2


def plot_dose_maps(axes_ref, dose_ref, dose_eval, gamma_vol, cfg, stats, out_dir, suffix=""):
    sl_ax  = _sl(dose_ref, 0, cfg["slice_axial"])
    sl_cor = _sl(dose_ref, 1, cfg["slice_coronal"])
    sl_sag = _sl(dose_ref, 2, cfg["slice_sagittal"])
    dmax   = max(dose_ref.max(), dose_eval.max())
    mode_lbl = cfg.get("_gamma_mode_label", "3D")

    fig = plt.figure(figsize=(20, 14), facecolor="#0d0d0d")
    fig.suptitle(
        f"{cfg['plan_name']}  —  Gamma {mode_lbl} "
        f"{cfg['dose_threshold_percent']}%/{cfg['distance_mm']}mm\n"
        f"Taux passage : {stats['pass_rate']:.1f}%  |  "
        f"γ moyen : {stats['mean']:.3f}  |  γ max : {stats['max']:.3f}  |  "
        f"γ P95 : {stats['p95']:.3f}",
        color="white", fontsize=11, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.08,
                           top=0.93, bottom=0.05, left=0.04, right=0.96)
    views = [
        ("Axiale",    dose_ref[sl_ax,:,:],  dose_eval[sl_ax,:,:],  gamma_vol[sl_ax,:,:]),
        ("Coronale",  dose_ref[:,sl_cor,:], dose_eval[:,sl_cor,:], gamma_vol[:,sl_cor,:]),
        ("Sagittale", dose_ref[:,:,sl_sag], dose_eval[:,:,sl_sag], gamma_vol[:,:,sl_sag]),
    ]

    for row, (label, sr, se, sg) in enumerate(views):
        diff = se - sr; dif_max = np.abs(diff).max() or 1e-6
        for col, (data, title, cmap, vmin, vmax) in enumerate([
            (sr,   f"{label} — {cfg['label_ref'][:16]}",  CMAP_DOSE, 0, dmax),
            (se,   f"{label} — {cfg['label_eval'][:16]}", CMAP_DOSE, 0, dmax),
            (diff, f"{label} — Différence (ΔGy)",          "RdBu_r", -dif_max, dif_max),
        ]):
            ax = fig.add_subplot(gs[row, col])
            im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax,
                           origin="lower", aspect="auto", interpolation="bilinear")
            ax.set_title(title, color="white", fontsize=7, pad=3)
            ax.set_facecolor("#0d0d0d")
            ax.tick_params(colors="gray", labelsize=5)
            for sp in ax.spines.values(): sp.set_edgecolor("#333")
            cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cb.ax.tick_params(colors="gray", labelsize=5)
            cb.set_label("Gy" if col < 2 else "ΔGy", color="gray", fontsize=6)

    for col, (label, _, _, sg) in enumerate(views):
        ax = fig.add_subplot(gs[3, col])
        im = ax.imshow(sg, cmap=CMAP_GAMMA, vmin=0, vmax=cfg["max_gamma"],
                       origin="lower", aspect="auto", interpolation="bilinear")
        ax.set_title(f"Gamma {mode_lbl} — {label}", color="white", fontsize=7, pad=3)
        ax.set_facecolor("#0d0d0d")
        ax.tick_params(colors="gray", labelsize=5)
        for sp in ax.spines.values(): sp.set_edgecolor("#333")
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(colors="gray", labelsize=5)
        cb.set_label("γ", color="gray", fontsize=7)

    fname = f"01_dose_maps{suffix}.png"
    path  = os.path.join(out_dir, fname)
    plt.savefig(path, dpi=cfg["dpi"], bbox_inches="tight", facecolor="#0d0d0d")
    plt.close()
    print(f"  ✓ Cartes de dose ({mode_lbl}) : {path}")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 : Histogramme gamma
# ─────────────────────────────────────────────────────────────────────────────

def plot_gamma_histogram(gamma_vol, cfg, stats, out_dir, suffix=""):
    valid    = gamma_vol[~np.isnan(gamma_vol)]
    mode_lbl = cfg.get("_gamma_mode_label", "3D")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), facecolor="#111")
    fig.suptitle(
        f"Analyse Gamma {mode_lbl} — {cfg['dose_threshold_percent']}%/{cfg['distance_mm']}mm  "
        f"(normalisation globale, seuil bas {cfg['lower_dose_cutoff']}% Dmax)\n"
        f"Réf : Low et al. Med Phys 1998 / AAPM TG-218",
        color="white", fontsize=11, fontweight="bold")

    bins   = np.linspace(0, cfg["max_gamma"], 80)
    counts, edges = np.histogram(valid, bins=bins)
    ctr    = 0.5*(edges[:-1]+edges[1:])
    cols_b = ["#2ecc71" if c <= 1.0 else "#e74c3c" for c in ctr]

    ax1.set_facecolor("#1a1a2e")
    ax1.bar(ctr, counts/len(valid)*100, width=edges[1]-edges[0],
            color=cols_b, alpha=0.85, edgecolor="none")
    ax1.axvline(1.0,           color="white",   lw=1.5, ls="--", label="γ=1")
    ax1.axvline(stats["mean"], color="#f39c12", lw=1.5, ls="-.", label=f"Moy : {stats['mean']:.3f}")
    ax1.axvline(stats["median"],color="#9b59b6",lw=1.5, ls=":",  label=f"Med : {stats['median']:.3f}")
    ax1.axvline(stats["p95"],  color="#e74c3c", lw=1,   ls=":",  label=f"P95 : {stats['p95']:.3f}")
    ax1.set_xlabel("γ", color="white"); ax1.set_ylabel("Fréquence (%)", color="white")
    ax1.set_title("Distribution différentielle", color="white")
    ax1.tick_params(colors="gray")
    ax1.legend(facecolor="#222", labelcolor="white", fontsize=8)
    for sp in ax1.spines.values(): sp.set_edgecolor("#333")
    pc = "#2ecc71" if stats["pass_rate"] >= 95 else "#e74c3c"
    ax1.text(0.98, 0.97, f"Taux passage : {stats['pass_rate']:.1f}%",
             transform=ax1.transAxes, ha="right", va="top", fontsize=13,
             fontweight="bold", color=pc,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#111", edgecolor=pc, alpha=0.9))

    ax2.set_facecolor("#1a1a2e")
    sv = np.sort(valid)
    cu = np.arange(1, len(sv)+1)/len(sv)*100
    ax2.plot(sv, cu, color="#3498db", lw=2.5)
    ax2.axvline(1.0, color="#e74c3c", lw=1.5, ls="--")
    ax2.axhline(stats["pass_rate"], color="#2ecc71", lw=1, ls="--", alpha=0.7,
                label=f"{stats['pass_rate']:.1f}% à γ≤1")
    ax2.set_xlabel("γ", color="white"); ax2.set_ylabel("Fraction cumulée (%)", color="white")
    ax2.set_title("Distribution cumulée", color="white")
    ax2.set_xlim(0, cfg["max_gamma"]); ax2.set_ylim(0, 101)
    ax2.tick_params(colors="gray")
    ax2.legend(facecolor="#222", labelcolor="white", fontsize=8)
    for sp in ax2.spines.values(): sp.set_edgecolor("#333")

    stat_lines = [
        ("Voxels évalués",   f"{stats['n_voxels_evaluated']:,}"),
        ("Taux passage γ≤1", f"{stats['pass_rate']:.2f} %"),
        ("γ moyen",          f"{stats['mean']:.4f}"),
        ("γ médian",         f"{stats['median']:.4f}"),
        ("γ max",            f"{stats['max']:.4f}"),
        ("γ P95",            f"{stats['p95']:.4f}"),
        ("Mode gamma",       mode_lbl),
        ("Critère dose",     f"{cfg['dose_threshold_percent']} %"),
        ("Critère dist.",    f"{cfg['distance_mm']} mm"),
        ("Seuil bas",        f"{cfg['lower_dose_cutoff']} % Dmax"),
        ("Normalisation",    "Globale (Dmax ref)"),
    ]
    yp = 0.97
    ax2.text(1.04, yp, "STATISTIQUES", transform=ax2.transAxes,
             color="#aaa", fontsize=8.5, fontweight="bold", va="top", fontfamily="monospace")
    for lbl, val in stat_lines:
        yp -= 0.082
        ax2.text(1.04, yp, f"{lbl:<22}{val}", transform=ax2.transAxes,
                 color="white", fontsize=7.5, va="top", fontfamily="monospace")

    plt.tight_layout(rect=[0,0,0.80,1])
    fname = f"02_gamma_histogram{suffix}.png"
    path  = os.path.join(out_dir, fname)
    plt.savefig(path, dpi=cfg["dpi"], bbox_inches="tight", facecolor="#111")
    plt.close()
    print(f"  ✓ Histogramme gamma ({mode_lbl}) : {path}")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 : DVH par structure
# Réf : AAPM TG-132
# ─────────────────────────────────────────────────────────────────────────────

def plot_dvh_by_roi(dose_ref, dose_eval, masks, cfg, out_dir):
    if not masks:
        masks = {"Volume global": np.ones(dose_ref.shape, dtype=bool)}

    dmax = max(dose_ref.max(), dose_eval.max()) * 1.02
    bins = np.linspace(0, dmax, 400)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 7), facecolor="#111")
    fig.suptitle(
        f"DVH par structure — {cfg['label_ref']} (—) vs {cfg['label_eval']} (- -)\n"
        f"Réf : AAPM TG-132",
        color="white", fontsize=12, fontweight="bold")

    for ax, title in [(ax1, "DVH Cumulé"), (ax2, "DVH Différentiel")]:
        ax.set_facecolor("#1a1a2e")
        ax.set_xlabel("Dose (Gy)", color="white", fontsize=10)
        ax.set_ylabel("Volume (%)", color="white", fontsize=10)
        ax.set_title(title, color="white", fontsize=11)
        ax.tick_params(colors="gray")
        ax.grid(color="#1e1e1e", lw=0.5)
        ax.set_xlim(0, dmax)
        for sp in ax.spines.values(): sp.set_edgecolor("#333")

    legend_handles = []
    for i, (roi_name, mask) in enumerate(masks.items()):
        col = PALETTE[i % len(PALETTE)]
        for dose_vol, ls, lw in [(dose_ref,"-",2.5),(dose_eval,"--",1.8)]:
            flat = dose_vol[mask].ravel()
            if len(flat) == 0:
                continue
            counts, edges = np.histogram(flat, bins=bins)
            ctr   = 0.5*(edges[:-1]+edges[1:])
            total = counts.sum()
            ax1.plot(ctr, 100*np.cumsum(counts[::-1])[::-1]/total,
                     color=col, lw=lw, ls=ls, alpha=0.9)
            ax2.plot(ctr, 100*counts/total,
                     color=col, lw=lw, ls=ls, alpha=0.85)
        legend_handles.append(mlines.Line2D([], [], color=col, lw=2.5, label=roi_name))

    legend_handles += [
        mlines.Line2D([], [], color="white", lw=2,   ls="-",  label=f"— {cfg['label_ref']}"),
        mlines.Line2D([], [], color="white", lw=1.5, ls="--", label=f"-- {cfg['label_eval']}"),
    ]
    for ax in [ax1, ax2]:
        ax.legend(handles=legend_handles, facecolor="#1a1a2e",
                  labelcolor="white", fontsize=7.5, loc="upper right", framealpha=0.8)

    plt.tight_layout()
    path = os.path.join(out_dir, "03_dvh_par_structure.png")
    plt.savefig(path, dpi=cfg["dpi"], bbox_inches="tight", facecolor="#111")
    plt.close()
    print(f"  ✓ DVH par structure : {path}")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 : Profils centraux
# ─────────────────────────────────────────────────────────────────────────────

def plot_dose_profiles(axes_ref, dose_ref, dose_eval, gamma_vol, cfg, out_dir, suffix=""):
    sz, sy, sx = dose_ref.shape[0]//2, dose_ref.shape[1]//2, dose_ref.shape[2]//2
    mode_lbl   = cfg.get("_gamma_mode_label", "3D")

    profiles = [
        ("Axe Z (SI)", dose_ref[:,sy,sx], dose_eval[:,sy,sx], gamma_vol[:,sy,sx]),
        ("Axe Y (AP)", dose_ref[sz,:,sx], dose_eval[sz,:,sx], gamma_vol[sz,:,sx]),
        ("Axe X (LR)", dose_ref[sz,sy,:], dose_eval[sz,sy,:], gamma_vol[sz,sy,:]),
    ]
    fig, grid = plt.subplots(3, 2, figsize=(16, 12), facecolor="#111")
    fig.suptitle(f"Profils de dose centraux + Gamma {mode_lbl}",
                 color="white", fontsize=13, fontweight="bold")

    for row, (lbl, pr, pe, pg) in enumerate(profiles):
        pos  = np.arange(len(pr))
        diff = pe - pr

        ax_d = grid[row, 0]; ax_d.set_facecolor("#1a1a2e")
        ax_d.plot(pos, pr, color="#3498db", lw=2, label=cfg["label_ref"])
        ax_d.plot(pos, pe, color="#e74c3c", lw=2, ls="--", label=cfg["label_eval"])
        ax_d.fill_between(pos, pr, pe, where=(diff>0), alpha=0.15, color="#e74c3c", label="Δ+")
        ax_d.fill_between(pos, pr, pe, where=(diff<0), alpha=0.15, color="#3498db", label="Δ−")
        ax_d.set_ylabel("Dose (Gy)", color="white", fontsize=9)
        ax_d.set_title(f"Profil {lbl}", color="white", fontsize=10)
        ax_d.tick_params(colors="gray", labelsize=8)
        ax_d.legend(facecolor="#222", labelcolor="white", fontsize=7)
        ax_d.grid(color="#1e1e1e")
        for sp in ax_d.spines.values(): sp.set_edgecolor("#333")

        ax_g = grid[row, 1]; ax_g.set_facecolor("#1a1a2e")
        vg   = np.where(np.isnan(pg), np.nan, pg)
        cpts = ["#2ecc71" if (not np.isnan(g) and g<=1) else "#e74c3c" for g in vg]
        ax_g.scatter(pos, vg, c=cpts, s=4, alpha=0.8, linewidths=0)
        ax_g.axhline(1.0, color="white", lw=1, ls="--", alpha=0.6)
        ax_g.set_ylim(0, cfg["max_gamma"])
        ax_g.set_ylabel("γ", color="white", fontsize=11)
        ax_g.set_title(f"Gamma {mode_lbl} — {lbl}", color="white", fontsize=10)
        ax_g.tick_params(colors="gray", labelsize=8)
        ax_g.grid(color="#1e1e1e")
        for sp in ax_g.spines.values(): sp.set_edgecolor("#333")
        vp = vg[~np.isnan(vg)] if not np.all(np.isnan(vg)) else np.array([])
        if len(vp) > 0:
            lp = 100*np.sum(vp<=1)/len(vp)
            ax_g.text(0.02, 0.96, f"Passage : {lp:.1f}%",
                      transform=ax_g.transAxes,
                      color="#2ecc71" if lp>=95 else "#e74c3c",
                      fontsize=9, fontweight="bold", va="top")

    plt.tight_layout(rect=[0,0,1,0.96])
    fname = f"04_dose_profiles{suffix}.png"
    path  = os.path.join(out_dir, fname)
    plt.savefig(path, dpi=cfg["dpi"], bbox_inches="tight", facecolor="#111")
    plt.close()
    print(f"  ✓ Profils de dose ({mode_lbl}) : {path}")

# ─────────────────────────────────────────────────────────────────────────────
# MÉTRIQUES DVH PAR ROI
# Réf : AAPM TG-132, ICRU 83
# ─────────────────────────────────────────────────────────────────────────────

def _dvp(data, pct):
    """Dose reçue par pct% du volume (Dxx%)."""
    if len(data) == 0:
        return np.nan
    return float(np.percentile(data, 100.0 - pct))

def _vdp(data, thr):
    """% volume recevant ≥ thr Gy (Vxx)."""
    if len(data) == 0:
        return np.nan
    return float(100.0 * np.sum(data >= thr) / len(data))


def compute_roi_metrics(dose_ref, dose_eval, masks, prescribed_dose_gy=None):
    """Métriques DVH pour chaque ROI.

    HI = (D2% - D98%) / Dp   (ICRU 83) — idéal = 0
    CI = V(Dp) / V_roi        (RTOG)    — idéal = 1
    """
    if not masks:
        print("  ⚠  Pas de RTSTRUCT — calcul sur volume global irradié")
        masks = {"Volume_irradié": dose_ref > 0.01}

    if prescribed_dose_gy is None:
        flat_all = dose_ref[dose_ref > 0.01].ravel()
        prescribed_dose_gy = float(np.percentile(flat_all, 98)) if len(flat_all) else 1.0
        print(f"     Dp auto (D98 global CT) : {prescribed_dose_gy:.2f} Gy")

    dp = prescribed_dose_gy
    results = {}

    dose_keys = ["Dmax","Dmean","Dmin","D98","D95","D50","D2"]
    vol_keys  = ["V95","V100","V105","V107"]
    idx_keys  = ["HI","CI"]

    for roi_name, mask in masks.items():
        ref_flat  = dose_ref[mask].ravel()
        eval_flat = dose_eval[mask].ravel()

        if len(ref_flat) == 0:
            print(f"  ⚠  '{roi_name}' vide — ignorée")
            continue

        def met(data):
            if len(data) == 0:
                return {k: np.nan for k in dose_keys+vol_keys+idx_keys}
            d2  = _dvp(data, 2.0)
            d98 = _dvp(data, 98.0)
            return {
                "Dmax":  float(np.max(data)),
                "Dmean": float(np.mean(data)),
                "Dmin":  float(np.min(data)),
                "D98":   d98,
                "D95":   _dvp(data, 95.0),
                "D50":   _dvp(data, 50.0),
                "D2":    d2,
                "V95":   _vdp(data, dp*0.95),
                "V100":  _vdp(data, dp),
                "V105":  _vdp(data, dp*1.05),
                "V107":  _vdp(data, dp*1.07),
                "HI":    float((d2 - d98) / dp) if dp > 0 else np.nan,
                "CI":    float(np.sum(data >= dp) / len(data)),
            }

        ct_m = met(ref_flat)
        cb_m = met(eval_flat)
        delta     = {}
        delta_pct = {}

        for m in dose_keys:
            d = cb_m[m] - ct_m[m]
            delta[m]     = d
            delta_pct[m] = 100*d/ct_m[m] if (ct_m[m] not in (0, np.nan) and not np.isnan(ct_m[m])) else np.nan

        for m in vol_keys + idx_keys:
            d = cb_m[m] - ct_m[m]
            delta[m]     = d
            delta_pct[m] = d   # déjà en % ou sans unité

        results[roi_name] = {
            "CT": ct_m, "CBCT": cb_m,
            "delta": delta, "delta_pct": delta_pct,
            "n_voxels": int(mask.sum()),
            "prescribed_dose_gy": dp,
        }
        print(f"     Métriques '{roi_name}' : Dmax CT={ct_m['Dmax']:.2f}Gy / CBCT={cb_m['Dmax']:.2f}Gy"
              f" | V100 CT={ct_m['V100']:.1f}% / CBCT={cb_m['V100']:.1f}%")

    return results, dp

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5 : Tableau métriques par ROI
# ─────────────────────────────────────────────────────────────────────────────

def plot_roi_metrics_table(all_roi_metrics, dp, cfg, out_dir):
    if not all_roi_metrics:
        print("  ⚠  Aucune ROI à afficher")
        return

    dose_keys = ["Dmax","Dmean","D98","D95","D50","D2"]
    vol_keys  = ["V95","V100","V105","V107"]
    idx_keys  = ["HI","CI"]
    all_keys  = dose_keys + vol_keys + idx_keys

    n_rois       = len(all_roi_metrics)
    rows_per_roi = len(all_keys) + 4
    fig_h        = max(9, 1.5 + n_rois * rows_per_roi * 0.38)
    fig, ax = plt.subplots(figsize=(17, fig_h), facecolor="#111")
    ax.set_facecolor("#111"); ax.axis("off")
    fig.suptitle(
        f"Métriques dosimétriques par structure — CT vs CBCT\n"
        f"Dp = {dp:.2f} Gy  |  HI=(D2−D98)/Dp (ICRU83, idéal=0)  |  "
        f"CI=V(Dp)/Vroi (RTOG, idéal=1)  |  Réf : AAPM TG-132",
        color="white", fontsize=10, fontweight="bold", y=0.99)

    xs   = [0.01, 0.13, 0.35, 0.50, 0.65, 0.78, 0.90]
    hdrs = ["Métrique","Description","CT","CBCT","Δ","Δ%","Status"]
    descs = {
        "Dmax":"Dose max","Dmean":"Dose moyenne",
        "D98":"D98% ← quasi-min","D95":"D95% ← couverture",
        "D50":"D50% médiane","D2":"D2% ← quasi-max",
        "V95":"Vol ≥ 95% Dp","V100":"Vol ≥ 100% Dp ← clé",
        "V105":"Vol ≥ 105% Dp","V107":"Vol ≥ 107% Dp",
        "HI":"HI=(D2-D98)/Dp  idéal=0","CI":"CI=V(Dp)/Vroi  idéal=1",
    }
    units = {m:"Gy" for m in dose_keys}
    units.update({m:"%" for m in vol_keys})
    units.update({m:"" for m in idx_keys})

    def st(v):
        if np.isnan(v): return "—"
        return "✓" if abs(v)<1 else ("⚠" if abs(v)<3 else "✗")

    def cst(v):
        if np.isnan(v): return "#666"
        return "#2ecc71" if abs(v)<1 else ("#f39c12" if abs(v)<3 else "#e74c3c")

    rh = min(0.042, 0.92/(n_rois*rows_per_roi+2))
    y  = 0.94

    for roi_name, roi_data in all_roi_metrics.items():
        ax.text(0.01, y, f"▶  {roi_name}  ({roi_data['n_voxels']:,} voxels)",
                transform=ax.transAxes, color="#f39c12", fontsize=9.5,
                fontweight="bold", va="top")
        y -= rh*0.9

        for xi, h in enumerate(hdrs):
            ax.text(xs[xi], y, h, transform=ax.transAxes,
                    color="#888", fontsize=7, fontweight="bold", va="top", fontfamily="monospace")
        y -= rh*0.2
        ax.plot([0.01,0.99],[y,y], transform=ax.transAxes, color="#333", lw=0.6)
        y -= rh*0.6

        for i, m in enumerate(all_keys):
            if m in ("V95", "HI"):
                ax.plot([0.01,0.99],[y+rh*0.75, y+rh*0.75],
                        transform=ax.transAxes, color="#222", lw=0.4)

            ct_v = roi_data["CT"][m]; cb_v = roi_data["CBCT"][m]
            dlt  = roi_data["delta"][m]; dp_v = roi_data["delta_pct"][m]
            u    = units[m]; col = cst(dp_v)
            bg   = "#0d0d18" if i%2==0 else "#111"
            ax.add_patch(plt.Rectangle((0, y-rh*0.2), 1, rh*0.9,
                         transform=ax.transAxes, color=bg, zorder=0))
            fmt = ".3f" if u == "Gy" else ".2f"
            ax.text(xs[0], y, m,                           transform=ax.transAxes, color="#3498db", fontsize=7, va="top", fontweight="bold", fontfamily="monospace")
            ax.text(xs[1], y, descs.get(m,""),             transform=ax.transAxes, color="#aaa",    fontsize=6.5, va="top")
            ax.text(xs[2], y, f"{ct_v:{fmt}}{u}",         transform=ax.transAxes, color="white",   fontsize=7, va="top", fontfamily="monospace")
            ax.text(xs[3], y, f"{cb_v:{fmt}}{u}",         transform=ax.transAxes, color="white",   fontsize=7, va="top", fontfamily="monospace")
            ax.text(xs[4], y, f"{dlt:+{fmt}}{u}",         transform=ax.transAxes, color=col,       fontsize=7, va="top", fontweight="bold", fontfamily="monospace")
            ax.text(xs[5], y, f"{dp_v:+.2f}%" if not np.isnan(dp_v) else "—",
                    transform=ax.transAxes, color=col, fontsize=7, va="top",
                    fontweight="bold", fontfamily="monospace")
            ax.text(xs[6], y, st(dp_v),                   transform=ax.transAxes, color=col,       fontsize=8, va="top", fontweight="bold")
            y -= rh

        y -= rh * 1.0

    # Légende
    y = min(y, 0.06)
    for col, lbl in [("#2ecc71","Δ < 1% — Excellent"),
                     ("#f39c12","1–3% — Acceptable"),
                     ("#e74c3c","≥ 3% — À investiguer")]:
        ax.add_patch(plt.Rectangle((0.01, y-0.009), 0.010, 0.016,
                     transform=ax.transAxes, color=col))
        ax.text(0.025, y, lbl, transform=ax.transAxes, color="#aaa", fontsize=7, va="top")
        y -= 0.022

    path = os.path.join(out_dir, "05_metrics_par_roi.png")
    plt.savefig(path, dpi=cfg["dpi"], bbox_inches="tight", facecolor="#111")
    plt.close()
    print(f"  ✓ Tableau métriques par ROI : {path}")

# ─────────────────────────────────────────────────────────────────────────────
# RAPPORT TEXTE COMPLET
# ─────────────────────────────────────────────────────────────────────────────

def write_report(cfg, all_stats, all_roi_metrics, dp, out_dir):
    path = os.path.join(out_dir, "00_rapport_complet.txt")
    dk = ["Dmax","Dmean","Dmin","D98","D95","D50","D2"]
    vk = ["V95","V100","V105","V107"]
    ik = ["HI","CI"]

    def st(v):
        if np.isnan(v): return "—"
        return "✓ Excellent" if abs(v)<1 else ("⚠ Acceptable" if abs(v)<3 else "✗ Attention")

    lines = [
        "="*72,
        "  RAPPORT DOSIMÉTRIQUE COMPLET — CT vs CBCT",
        "  Réf : Low et al. 1998, AAPM TG-218, ICRU 83, AAPM TG-132",
        "="*72,
        f"  Plan          : {cfg['plan_name']}",
        f"  Référence     : {cfg['label_ref']}",
        f"  Évaluation    : {cfg['label_eval']}",
        f"  Dp            : {dp:.2f} Gy",
        f"  Mode gamma    : {cfg['gamma_mode'].upper()}",
        f"  Critères      : {cfg['dose_threshold_percent']}% / {cfg['distance_mm']}mm",
        f"  Seuil bas     : {cfg['lower_dose_cutoff']}% Dmax",
        f"  Normalisation : Globale (Dmax référence)",
        "",
    ]

    for mode_label, stats in all_stats.items():
        conformite = "✓ CONFORME" if stats["pass_rate"] >= 95 else "✗ NON CONFORME"
        lines += [
            f"  ── GAMMA {mode_label} ────────────────────────────────────────────",
            f"  Voxels évalués : {stats['n_voxels_evaluated']:,}",
            f"  Taux passage   : {stats['pass_rate']:.2f}%  ({conformite}, seuil 95%)",
            f"  γ moyen        : {stats['mean']:.4f}",
            f"  γ médian       : {stats['median']:.4f}",
            f"  γ max          : {stats['max']:.4f}",
            f"  γ P95          : {stats['p95']:.4f}",
            "",
        ]

    for roi_name, roi_data in all_roi_metrics.items():
        lines += [
            f"  ── STRUCTURE : {roi_name}  ({roi_data['n_voxels']:,} voxels) ──────────",
            f"  {'Métrique':<8} {'CT':>10} {'CBCT':>10} {'Δ':>10} {'Δ%':>9}  Status",
            "  " + "-"*62,
        ]
        for grp, ml in [("Doses", dk), ("Volumes", vk), ("Indices", ik)]:
            lines.append(f"  [{grp}]")
            for m in ml:
                u = "Gy" if m in dk else ("%" if m in vk else "")
                ct_v = roi_data["CT"][m]; cb_v = roi_data["CBCT"][m]
                dlt  = roi_data["delta"][m]; dpv  = roi_data["delta_pct"][m]
                lines.append(
                    f"  {m:<8} {ct_v:>10.3f}{u:2} {cb_v:>10.3f}{u:2} "
                    f"{dlt:>+10.3f}{u:2} {dpv:>+8.2f}%  {st(dpv)}")
        lines.append("")

    lines += ["="*72]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  ✓ Rapport complet : {path}")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "═"*62)
    print("  GAMMA — Comparaison dosimétrique Monaco (CT vs CBCT)")
    print("  Réf : Low 1998, AAPM TG-218/132, ICRU 83")
    print("═"*62)
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    # 1. Chargement RT Dose
    print("\n[1/6] Chargement des RT Dose...")
    axes_ref,  dose_ref,  _ = load_rtdose(CONFIG["dose_reference"])
    axes_eval, dose_eval, _ = load_rtdose(CONFIG["dose_evaluation"])

    # 2. Alignement des grilles
    # ─────────────────────────────────────────────────────────────────────────
    # Cas couverts :
    #   A) Grilles strictement identiques (même origine, spacing, nb coupes) → rien à faire
    #   B) Même spacing mais origines décalées (isocentres différents)       → resamplage
    #   C) Nb de coupes différent (épaisseurs de coupe différentes)          → resamplage
    #   D) Spacing différent (résolution différente X/Y)                     → resamplage
    # Le resamplage par interpolation bilinéaire gère tous ces cas via
    # RegularGridInterpolator, avec correction du décalage isocentre.
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[2/6] Vérification et alignement des grilles...")

    # Toujours charger les isocentres (nécessaires aussi pour la correction RTSTRUCT)
    iso_r = get_isocenter(CONFIG.get("rtplan_reference"))
    iso_e = get_isocenter(CONFIG.get("rtplan_evaluation"))

    def _grilles_identiques(axes_ref, axes_eval):
        """Retourne True ssi les deux grilles sont strictement superposables."""
        for ar, ae in zip(axes_ref, axes_eval):
            if len(ar) != len(ae):
                return False, "nombre de voxels différent"
            if not np.allclose(ar, ae, atol=0.1):
                return False, "origines ou spacings différents"
        return True, "OK"

    ok, raison = _grilles_identiques(axes_ref, axes_eval)

    if ok:
        print("  ✓ Grilles parfaitement compatibles — pas de resamplage nécessaire")
    else:
        print(f"  ⚠  Grilles incompatibles ({raison}) — resamplage en cours...")
        for i, (ar, ae) in enumerate(zip(axes_ref, axes_eval)):
            sp_r = round(float(ar[1]-ar[0]), 3) if len(ar)>1 else 0
            sp_e = round(float(ae[1]-ae[0]), 3) if len(ae)>1 else 0
            print(f"     Axe {['Z','Y','X'][i]} : ref={len(ar)} pts "
                  f"[{ar[0]:.1f}…{ar[-1]:.1f}] Δ={sp_r}mm  |  "
                  f"eval={len(ae)} pts [{ae[0]:.1f}…{ae[-1]:.1f}] Δ={sp_e}mm")
        dose_eval = resample_eval_on_ref(axes_ref, axes_eval, dose_eval, iso_r, iso_e)
        axes_eval = axes_ref
        print("  ✓ Resamplage terminé")

    # 3. Chargement RTSTRUCT (avec correction isocentre pour les contours)
    print("\n[3/6] Chargement des structures (RTSTRUCT)...")
    masks = load_rtstruct_masks(
        CONFIG.get("rtss_file"),
        axes_ref,
        iso_ref=iso_r,
        iso_ss=iso_e,
        exclude=CONFIG.get("roi_exclude", []))

    # 4. Calcul gamma
    print("\n[4/6] Calcul Gamma...")
    mode     = CONFIG["gamma_mode"].lower()
    all_stats = {}

    if mode in ("3d", "both"):
        print("  ── Mode 3D volumique (AAPM TG-218) ──")
        CONFIG["_gamma_mode_label"] = "3D"
        gv3d = compute_gamma_3d(axes_ref, dose_ref, axes_eval, dose_eval, CONFIG)
        st3d = gamma_stats(gv3d)
        all_stats["3D"] = st3d
        print(f"  Passage : {st3d['pass_rate']:.2f}% | "
              f"γ moy : {st3d['mean']:.4f} | γ max : {st3d['max']:.4f}")
        plot_dose_maps(axes_ref, dose_ref, dose_eval, gv3d, CONFIG, st3d,
                       CONFIG["output_dir"], "_3D")
        plot_gamma_histogram(gv3d, CONFIG, st3d, CONFIG["output_dir"], "_3D")
        plot_dose_profiles(axes_ref, dose_ref, dose_eval, gv3d, CONFIG,
                           CONFIG["output_dir"], "_3D")

    if mode in ("2d", "both"):
        print("  ── Mode 2D plan par plan (style Doselab) ──")
        CONFIG["_gamma_mode_label"] = "2D"
        gv2d = compute_gamma_2d(axes_ref, dose_ref, axes_eval, dose_eval, CONFIG)
        st2d = gamma_stats(gv2d)
        all_stats["2D"] = st2d
        print(f"  Passage : {st2d['pass_rate']:.2f}% | "
              f"γ moy : {st2d['mean']:.4f} | γ max : {st2d['max']:.4f}")
        plot_dose_maps(axes_ref, dose_ref, dose_eval, gv2d, CONFIG, st2d,
                       CONFIG["output_dir"], "_2D")
        plot_gamma_histogram(gv2d, CONFIG, st2d, CONFIG["output_dir"], "_2D")
        plot_dose_profiles(axes_ref, dose_ref, dose_eval, gv2d, CONFIG,
                           CONFIG["output_dir"], "_2D")

    # 5. DVH + métriques par ROI
    print("\n[5/6] DVH et métriques par structure...")
    plot_dvh_by_roi(dose_ref, dose_eval, masks, CONFIG, CONFIG["output_dir"])
    all_roi_metrics, dp = compute_roi_metrics(
        dose_ref, dose_eval, masks, CONFIG.get("prescribed_dose_gy"))
    plot_roi_metrics_table(all_roi_metrics, dp, CONFIG, CONFIG["output_dir"])

    # 6. Rapport texte
    print("\n[6/6] Rapport complet...")
    write_report(CONFIG, all_stats, all_roi_metrics, dp, CONFIG["output_dir"])

    print("\n" + "═"*62)
    print(f"  ✅ Terminé ! Résultats dans : {CONFIG['output_dir']}/")
    print("═"*62 + "\n")


if __name__ == "__main__":
    main()
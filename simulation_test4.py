import math
import requests
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import CubicSpline

########################
# 1) GLOBAL CONSTANTS  #
########################
TOTAL_FRAMES = 145         # ~12 hours at 5-min increments
INTERVAL_MS = 200          # ms per frame
FRAMES_PER_HOUR = 12       # each hour has 12 frames if each frame=5min

AREA_SIZE_KM = 10.0
IMAGE_PIXELS = 500
AXIS_MARGIN = 50
SOLAR_PANEL_SIZE = 20
NUM_CLOUDS = 5
COVERAGE_THRESHOLD = 0.2

BACKGROUND_COLOR = (102, 204, 102)
PANEL_COLOR = (50, 50, 150)
CLOUD_COLOR = (200, 200, 200, 180)  # <-- Important to define this here

# Minimal placeholder baseline arrays for each panel
custom_baseline_A = np.array([0.1, 0.2, 0.3])
custom_baseline_D = np.array([0.1, 0.2, 0.3])
custom_baseline_K = np.array([0.1, 0.2, 0.3])
custom_baseline_J = np.array([0.1, 0.2, 0.3])
custom_baseline_L = np.array([0.1, 0.2, 0.3])
custom_baseline_G = np.array([0.1, 0.2, 0.3])
baseline_data_common = np.array([0.1, 0.2, 0.3])

panel_names = list("ABCDEFGHIJKLMNOPQRST")  # up to 20
default_coords = [
    (8.3, 4.2), (1.6, 9.8), (6.7, 5.1), (9.2, 0.8), (4.5, 7.1),
    (7.3, 8.6), (0.6, 6.3), (3.9, 2.0), (5.8, 8.4), (3.1, 4.9),
    (8.2, 9.9), (5.7, 5.6), (2.9, 0.3), (6.4, 7.8), (9.3, 3.9),
    (5.1, 6.9), (7.0, 9.2), (1.2, 4.6), (9.5, 2.5), (5.1, 1.2),
]
custom_coords = {
    "A": (8.3, 4.2), "D": (9.2, 0.8), "G": (0.6, 6.3),
    "J": (3.9, 2.0), "K": (0.6, 6.3), "L": (3.1, 4.9)
}

panels = []
for i, pname in enumerate(panel_names):
    if i>=len(default_coords):
        break
    coord = custom_coords.get(pname, default_coords[i])
    panels.append({'name': pname, 'x_km': coord[0], 'y_km': coord[1]})

custom_cloud_factors = {"A": 0.7, "D": 0.65, "K": 0.55, "J": 0.6, "L": 0.6, "G": 0.55}
cloud_factor_dict = {}
for pname in panel_names:
    cloud_factor_dict[pname] = custom_cloud_factors.get(pname, 0.5)

# Build baseline arrays for each panel
time_idx = np.linspace(0, TOTAL_FRAMES - 1, TOTAL_FRAMES)
baseline_data_dict = {}
for pname in panel_names:
    if pname=="A":
        arr = custom_baseline_A
    elif pname=="D":
        arr = custom_baseline_D
    elif pname=="K":
        arr = custom_baseline_K
    elif pname=="J":
        arr = custom_baseline_J
    elif pname=="L":
        arr = custom_baseline_L
    elif pname=="G":
        arr = custom_baseline_G
    else:
        arr = baseline_data_common
    if len(arr)!=TOTAL_FRAMES:
        x_old = np.linspace(0,len(arr)-1,len(arr))
        x_new = np.linspace(0,len(arr)-1,TOTAL_FRAMES)
        arr = np.interp(x_new,x_old,arr)
    # store as a function for convenience
    baseline_data_dict[pname] = ( 
        lambda A: lambda x: np.interp(x, range(len(A)), A) 
    )(arr)

########################
# 2) Weather API
########################
def get_weather_data(date_str, lat=6.9355, lon=79.8487):
    
    
    base_url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "cloudcover,windspeed_10m,winddirection_10m",
        "start_date": date_str,
        "end_date": date_str,
        "timezone": "auto"
    }
    try:
        resp = requests.get(base_url, params=params, timeout=10)
        data = resp.json()
        cc = data["hourly"]["cloudcover"]         # array of 24 values
        wspd = data["hourly"]["windspeed_10m"]    # array of 24 (m/s)
        wdir = data["hourly"]["winddirection_10m"]# array of 24 (deg)
        return cc, wspd, wdir
    except Exception as e:
        print("API error, fallback => 50% CC, 5 m/s, 270 deg")
        return [50.0]*24, [5.0]*24, [270.0]*24

########################
# 3) Cloud creation & movement
########################
def create_clouds():
    clouds=[]
    for _ in range(NUM_CLOUDS):
        w = int(np.random.rand()*80+20)
        h = int(np.random.rand()*50+20)
        x_init = np.random.randint(0, IMAGE_PIXELS - w)
        y_init = np.random.randint(0, IMAGE_PIXELS - h)
        clouds.append({
            'x': x_init,
            'y': y_init,
            'width': w,
            'height': h
        })
    return clouds

def move_clouds(clouds, frame_idx, wind_speed_24h, wind_dir_24h):
    hour = frame_idx // FRAMES_PER_HOUR
    if hour>=24:
        hour=23
    ws_m_s=wind_speed_24h[hour]
    wd_deg=wind_dir_24h[hour]
    wd_rad=math.radians(wd_deg)
    dist_km = ws_m_s*0.06*5  # m/s => km/min => x5 => distance in 5 min
    px_per_km = IMAGE_PIXELS/AREA_SIZE_KM
    dx_px = dist_km*math.cos(wd_rad)*px_per_km
    dy_px = -dist_km*math.sin(wd_rad)*px_per_km
    for c in clouds:
        c['x'] += dx_px
        c['y'] += dy_px
        # wrap around
        if c['x']>IMAGE_PIXELS + c['width']:
            c['x']=-c['width']
        elif c['x']< -c['width']:
            c['x']=IMAGE_PIXELS
        if c['y']>IMAGE_PIXELS + c['height']:
            c['y']=-c['height']
        elif c['y']< -c['height']:
            c['y']=IMAGE_PIXELS

########################
# 4) coverage ratio
########################
def coverage_ratio_with_cloudcover(panel_box, cloud_box, cc_percent):
    px1, py1, px2, py2 = panel_box
    cx1, cy1, cx2, cy2 = cloud_box
    overlap_x = max(0, min(px2, cx2) - max(px1, cx1))
    overlap_y = max(0, min(py2, cy2) - max(py1, cy1))
    overlap_area=overlap_x*overlap_y
    panel_area=(px2-px1)*(py2-py1)
    raw_ratio=0.0
    if panel_area>0:
        raw_ratio= overlap_area/panel_area
    # scale by cloud cover
    scaled_ratio= raw_ratio*(cc_percent/100.0)
    return scaled_ratio

########################
# 5) Map drawing
########################
def create_base_map():
    base=Image.new("RGB",(IMAGE_PIXELS,IMAGE_PIXELS),BACKGROUND_COLOR)
    d=ImageDraw.Draw(base)
    axis_length=IMAGE_PIXELS-2*AXIS_MARGIN
    x0=AXIS_MARGIN
    y0=AXIS_MARGIN
    x1=x0+axis_length
    y1=y0+axis_length
    # draw bounding axes
    d.line([(x0,y1),(x1,y1)],fill=(0,0,0))
    d.line([(x0,y0),(x0,y1)],fill=(0,0,0))
    # x ticks
    for i in range(int(AREA_SIZE_KM)+1):
        px=x0+int((i/AREA_SIZE_KM)*axis_length)
        d.line([(px,y1),(px,y1+5)],fill=(0,0,0))
        d.text((px,y1+7),f"{i}",fill=(0,0,0))
    # y ticks
    for i in range(int(AREA_SIZE_KM)+1):
        py=y1-int((i/AREA_SIZE_KM)*axis_length)
        d.line([(x0-5,py),(x0,py)],fill=(0,0,0))
        d.text((x0-25,py),f"{i}",fill=(0,0,0))

    # panels
    try:
        font=ImageFont.truetype("arial.ttf",12)
    except:
        font=ImageFont.load_default()

    for p in panels:
        px_km=p['x_km']
        py_km=p['y_km']
        px=x0+int((px_km/AREA_SIZE_KM)*axis_length)
        py=y1-int((py_km/AREA_SIZE_KM)*axis_length)
        box=(px-SOLAR_PANEL_SIZE//2, py-SOLAR_PANEL_SIZE//2,
             px+SOLAR_PANEL_SIZE//2, py+SOLAR_PANEL_SIZE//2)
        d.rectangle(box, fill=PANEL_COLOR)
        d.text((px,py-SOLAR_PANEL_SIZE//2-15), p['name'], fill=(255,255,255),font=font)

    return base

def draw_clouds(base_img, clouds):
    overlay=Image.new("RGBA",(IMAGE_PIXELS,IMAGE_PIXELS),(0,0,0,0))
    dd=ImageDraw.Draw(overlay)
    for c in clouds:
        c_box=(c['x'],c['y'], c['x']+c['width'], c['y']+c['height'])
        dd.ellipse(c_box, fill=CLOUD_COLOR)
    base_img.paste(overlay,(0,0), overlay)

def get_time_string(frame_idx):
    # 5-min intervals starting ~06:20
    minutes=6*60+20 + frame_idx*5
    hh=minutes//60
    mm=minutes%60
    return f"{hh:02d}:{mm:02d}"

########################
# 6) MAIN
########################
def main():
    # pick date
    date_str="2025-04-10"
    # fetch from openmeteo
    cc_24, wspd_24, wdir_24= get_weather_data(date_str)

    base_map= create_base_map()
    clouds= create_clouds()

    fig, ax=plt.subplots()
    plt.axis("off")
    im=ax.imshow(np.array(base_map), animated=True)

    coverage_start_times={}

    def update(frame_idx):
        # figure out hour
        hour=frame_idx//FRAMES_PER_HOUR
        if hour>23:
            hour=23
        cc_percent= cc_24[hour]
        ws_m_s= wspd_24[hour]
        wd_deg= wdir_24[hour]

        # move clouds
        move_clouds(clouds, frame_idx, wspd_24, wdir_24)

        # copy map and draw
        frame_img= base_map.copy()
        draw_clouds(frame_img, clouds)

        # coverage/generation
        axis_length= IMAGE_PIXELS-2*AXIS_MARGIN
        x0=AXIS_MARGIN
        y1=x0+axis_length
        panel_info={}
        for p in panels:
            pname= p['name']
            baseline_fn= baseline_data_dict[pname]
            baseline_val= baseline_fn(frame_idx)
            px= x0+ int((p['x_km']/AREA_SIZE_KM)*axis_length)
            py= y1- int((p['y_km']/AREA_SIZE_KM)*axis_length)
            box_p=(px-SOLAR_PANEL_SIZE//2, py-SOLAR_PANEL_SIZE//2,
                   px+SOLAR_PANEL_SIZE//2, py+SOLAR_PANEL_SIZE//2)
            max_cov=0.0
            for c in clouds:
                c_box=( c['x'], c['y'], c['x']+c['width'], c['y']+c['height'] )
                cov= coverage_ratio_with_cloudcover(box_p, c_box, cc_percent)
                if cov>max_cov:
                    max_cov=cov
            cf= cloud_factor_dict[pname]
            eff_red= max_cov*cf
            if eff_red>1.0:
                eff_red=1.0
            final_gen= baseline_val*(1-eff_red)
            panel_info[pname]= final_gen

        total_gen= sum(panel_info.values())
        timestr= get_time_string(frame_idx)
        from PIL import ImageDraw
        d= ImageDraw.Draw(frame_img)
        txt=(
            f"Time: {timestr}\n"
            f"Hour: {hour}\n"
            f"CloudCover: {cc_percent:.0f}%\n"
            f"Wind: {ws_m_s:.1f} m/s @ {wd_deg:.0f}°\n"
            f"Total Gen: {total_gen:.1f}"
        )
        d.text((10,10), txt, fill=(255,0,0))

        print(f"Frame={frame_idx}, hr={hour}, CloudCover={cc_percent}%, Wind={ws_m_s}m/s dir={wd_deg}, TotGen={total_gen:.1f}")
        im.set_data(np.array(frame_img))
        return [im]

    # remove blit=True => to avoid the _resize_id error
    ani= FuncAnimation(fig, update, frames=range(TOTAL_FRAMES), interval=INTERVAL_MS, blit=False)
    plt.show()

if __name__=="__main__":
    main()

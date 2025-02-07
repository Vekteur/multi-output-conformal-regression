"""
This file is based on https://github.com/Zhendong-Wang/Probabilistic-Conformal-Prediction/blob/main/pcp/vis.py
"""

from pathlib import Path
from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt
import torch
import folium
import folium.features


Path('./taxi_example/figs/').mkdir(exist_ok=True)
figs_path = Path('./taxi_example/figs/taxi/')
figs_path.mkdir(exist_ok=True)


def get_angle(p1, p2):
    """
    This function Returns angle value in degree from the location p1 to location p2. Please
    refer to the following link for better understanding : https://gist.github.com/jeromer/2005586

    Args:
        - p1 : namedtuple with lat lon
        - p2 : namedtuple with lat lon

    Returns:
        - angle (float) : angle in degree
    """
    longitude_diff = np.radians(p2.lon - p1.lon)

    latitude1 = np.radians(p1.lat)
    latitude2 = np.radians(p2.lat)

    x_vector = np.sin(longitude_diff) * np.cos(latitude2)
    y_vector = np.cos(latitude1) * np.sin(latitude2) - (np.sin(latitude1) * np.cos(latitude2) * np.cos(longitude_diff))
    angle = np.degrees(np.arctan2(x_vector, y_vector))

    # Checking and adjustring angle value on the scale of 360
    if angle < 0:
        return angle + 360
    return angle


def getArrows(locations, color='red', size=6, n_arrows=3):
    """
    Get a list of placed and rotated arrows or markers to be plotted

    Args:
        - locations : list of lists of latitude longitude that represent the begining and end of Line.
                    this function Return list of arrows or the markers
        - color : color of the arrow
        - size : size of the arrow
        - n_arrows : number of arrows to be plotted
    """
    Point = namedtuple('Point', field_names=['lat', 'lon'])

    # creating point from Point named tuple
    point1 = Point(locations[0][0], locations[0][1])
    point2 = Point(locations[1][0], locations[1][1])

    # calculate the rotation required for the marker.
    # Reducing 90 to account for the orientation of marker
    # Get the degree of rotation
    angle = get_angle(point1, point2) - 90

    # get the evenly space list of latitudes and longitudes for the required arrows

    arrow_latitude = np.linspace(point1.lat, point2.lat, n_arrows + 2)[1 : n_arrows + 1]
    arrow_longitude = np.linspace(point1.lon, point2.lon, n_arrows + 2)[1 : n_arrows + 1]

    final_arrows = []

    color = 'red'

    # creating each "arrow" and appending them to our arrows list
    for points in zip(arrow_latitude, arrow_longitude):
        final_arrows.append(
            folium.RegularPolygonMarker(
                location=points, color=color, fill_color=color, number_of_sides=3, radius=size, rotation=angle
            )
        )
    return final_arrows


def visualize_data_on_map(X_test, Y_test, img_idx):
    """
    This function is used to visualize the data on the map. It will plot the pickup and dropoff location

    Args:
        - X_test : numpy array of pickup location
        - Y_test : numpy array of dropoff location
        - img_idx : index of the image
    """
    new_york_lat, new_york_lon = 40.730610, -73.935242
    map_1 = folium.Map(location=[new_york_lat, new_york_lon], tiles='OpenStreetMap', zoom_start=12)
    for each in range(len(X_test)):
        pickup_lat, pickup_lon = X_test[each, 0].item(), X_test[each, 1].item()
        dropoff_lat, dropoff_lon = Y_test[each, 0].item(), Y_test[each, 1].item()

        folium.Marker(
            [pickup_lon, pickup_lat], popup=str(each) + ': ' + 'pickup', icon=folium.Icon(color='red')
        ).add_to(map_1)
        folium.Marker(
            [dropoff_lon, dropoff_lat], popup=str(each) + ': ' + 'dropoff', icon=folium.Icon(color='blue')
        ).add_to(map_1)
        poly = folium.PolyLine(
            [[pickup_lon, pickup_lat], [dropoff_lon, dropoff_lat]], color="red", weight=2.5, opacity=1, stroke=True
        )
        poly.add_to(map_1)
        arrows = getArrows(locations=[[pickup_lon, pickup_lat], [dropoff_lon, dropoff_lat]], color='red', n_arrows=1)
        for arrow in arrows:
            arrow.add_to(map_1)

    # Save the map
    save_path = figs_path / f'{img_idx}'
    save_path.mkdir(exist_ok=True)

    map_1.save(str(save_path / 'data_map.html'))


def get_contour(x_value, conformalizer, alpha, ylim, zlim, grid_side=100, cache={}):
    """
    This function is used to get the contour of the region on the map

    Args:
        - x_value : value of the x-axis
        - conformalizer : conformalizer object
        - alpha : alpha value
        - ylim : y-axis limit
        - zlim : z-axis limit
        - grid_side : grid size
        - cache : cache object

    Returns:
        - contour.collections[0].get_paths() : contour paths
    """
    device = conformalizer.model.device
    y1, y2 = torch.linspace(*ylim, grid_side, device=device), torch.linspace(*zlim, grid_side, device=device)
    Y1, Y2 = torch.meshgrid(y1, y2, indexing='ij')
    pos = torch.dstack((Y1, Y2))
    pos = pos[:, :, None, :]
    assert pos.shape == (y1.shape[0], y1.shape[0], 1, 2)
    mask = conformalizer.is_in_region(x_value, pos, alpha, cache=cache)
    mask = mask[:, :, 0]

    Y1, Y2, mask = Y1.cpu().numpy(), Y2.cpu().numpy(), mask.float().cpu().numpy()
    assert Y1.shape == Y2.shape == mask.shape

    fig2D, ax2D = plt.subplots()
    contour = ax2D.contourf(Y1.T, Y2.T, mask.T, levels=2, colors='r')
    plt.close(fig2D)

    return contour.collections[0].get_paths()


def show_contours_on_map(conformalizer_name, x_test, y_test, contour_paths, img_idx, region_size, datamodule):
    """
    This function is used to show the map with the contour of the region

    Args:
        - conformalizer_name : conformalizer name
        - x_test : x-axis value
        - y_test : y-axis value
        - contour_paths : contour paths
        - img_idx : image index
        - region_size : region size
        - datamodule : datamodule object
    """
    pickup_lat, pickup_lon = x_test[0], x_test[1]
    dropoff_lat, dropoff_lon = y_test[0], y_test[1]

    # Plot the point
    new_york_lat, new_york_lon = 40.730610, -73.935242
    map_1 = folium.Map(location=[new_york_lat, new_york_lon], tiles='OpenStreetMap', zoom_start=12)

    # Plot the contour
    if len(contour_paths) > 0:
        contour_path = contour_paths[0]
        final_contour = []
        for j, contour_points in enumerate(contour_path.to_polygons()):
            if j == 0:
                continue

            parsed_points = datamodule.scaler_y.inverse_transform(contour_points).numpy()
            parsed_points = parsed_points[:, [1, 0]]

            final_contour.append(parsed_points)

        folium.Polygon(
            locations=final_contour, color='#FFFF00', fill=False, fill_color='#3b3b3b', weight=2, opacity=0.8
        ).add_to(map_1)

    folium.Marker([pickup_lon, pickup_lat], popup='pickup', icon=folium.Icon(color='red')).add_to(map_1)
    folium.Marker([y_test[1], y_test[0]], popup='dropoff', icon=folium.Icon(color='blue')).add_to(map_1)
    folium.PolyLine([[pickup_lon, pickup_lat], [y_test[1], y_test[0]]], color="red", weight=2.5, opacity=1).add_to(
        map_1
    )

    # Write the region size
    folium.map.Marker(
        [40.638713, -74.164311],
        icon=folium.features.DivIcon(
            icon_size=(320, 80),
            icon_anchor=(0, 0),
            html=f'<div style="display: flex; align-items: center; justify-content: center; font-size: 40pt; color: black; background-color: white; padding: 5px 10px; border-radius: 10px">Size: {round(region_size, 2)}</div>',
        ),
    ).add_to(map_1)

    # Save the map
    save_path = figs_path / f'{img_idx}'
    save_path.mkdir(exist_ok=True)

    map_1.save(str(save_path / f'{conformalizer_name.lower()}_map.html'))

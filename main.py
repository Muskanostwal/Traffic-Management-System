import cv2
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from ultralytics import YOLO
from tracker import Tracker  
import cv2.bgsegm
import winsound  

# Initialize YOLO model
model = YOLO('yolov8s.pt')

# Background Subtractor for density analysis
algo = cv2.bgsegm.createBackgroundSubtractorMOG()

# Tracker for vehicle tracking
tracker = Tracker()

# Video capture
cap = cv2.VideoCapture('video_vehicles.mp4')

# Load class names for detection (ensure the file exists)
with open("cocoa.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# Vehicle counters and constants
cy1, cy2 = 322, 368  # Y-coordinates for counting lines
offset = 6  # Offset for accuracy in counting
counter_up, counter_down = 0, 0  # Counters for up and down traffic
vehicle_crossed = {}  # To track which vehicles have crossed the lines
car_count, truck_count = 0, 0  # Initialize vehicle counts
car_speeds, truck_speeds = [], []  # Lists to hold speed data for analysis
VEHICLE_COUNT_THRESHOLD = 15  # Threshold for alert

# Variables for pie chart and density analysis
start_time = time.time()
x_density_data, y_density_data = [], []
x_speed_data, y_speed_data_car, y_speed_data_truck = [], [], []

# Set up live plotting (matplotlib)
plt.ion()
fig, (ax_pie, ax_density, ax_speed) = plt.subplots(1, 3, figsize=(12, 4))

# Function to trigger an alert sound
def alert_sound():
    # Play a beep sound (Windows-specific)
    winsound.Beep(1000, 500)  # Frequency = 2000 Hz, Duration = 5s

# Function to calculate vehicle speed (based on movement between frames)
def calculate_speed(vehicle_id, cx, cy, timestamp):
    # Placeholder logic for speed calculation (implement as needed)
    return np.random.uniform(20, 60)  # Simulating random speed between 20 and 60 km/h

# Main processing loop
count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    count += 1
    if count % 3 != 0:  # Process every third frame to optimize performance
        continue

    frame = cv2.resize(frame, (1020, 500))

    # Vehicle detection using YOLOv8
    results = model.predict(frame)
    boxes = results[0].boxes.data
    if len(boxes) == 0:
        continue

    px = pd.DataFrame(boxes).astype("float")

    vehicle_list = []
    timestamp = time.time()  # Current time for speed calculation
    for _, row in px.iterrows():
        x1, y1, x2, y2, class_id = int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[5])
        class_name = class_list[class_id]

        if 'car' in class_name or 'truck' in class_name:  # Track cars and trucks
            vehicle_list.append([x1, y1, x2, y2])
            speed = calculate_speed(_, (x1 + x2) // 2, (y1 + y2) // 2, timestamp)

            if speed:  # If speed is calculated, store it
                if 'car' in class_name:
                    car_speeds.append(speed)
                elif 'truck' in class_name:
                    truck_speeds.append(speed)

        # Count the vehicles by class
        if 'car' in class_name:
            car_count += 1
        elif 'truck' in class_name:
            truck_count += 1

    # Update tracker with detected vehicles
    bbox_id = tracker.update(vehicle_list)

    for bbox in bbox_id:
        x3, y3, x4, y4, vehicle_id = bbox
        cx, cy = (x3 + x4) // 2, (y3 + y4) // 2

        # Draw bounding box, ID, and center point
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(frame, str(vehicle_id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0, (255, 255, 255), 2)

        # Draw counting lines
        cv2.line(frame, (274, cy1), (814, cy1), (255, 0, 0), 2)
        cv2.line(frame, (177, cy2), (927, cy2), (0, 255, 0), 2)

        # Vehicle counting logic
        if cy1 - offset <= cy <= cy1 + offset:
            if vehicle_id not in vehicle_crossed:
                counter_down += 1
                vehicle_crossed[vehicle_id] = 'down'
        elif cy2 - offset <= cy <= cy2 + offset:
            if vehicle_id not in vehicle_crossed:
                counter_up += 1
                vehicle_crossed[vehicle_id] = 'up'

    # Vehicle counting display on frame
    cv2.putText(frame, "Down Count: " + str(counter_down), (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(frame, "Up Count: " + str(counter_up), (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    

    # Update pie chart and density analysis every minute
    elapsed_time = time.time() - start_time
    if elapsed_time >= 30:
        # Total vehicle count in the last minute
        total_vehicles = car_count + truck_count
        cv2.putText(frame, "car count: " + str(car_count), (1000, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        

        # Trigger alert if vehicle count exceeds the threshold
        if total_vehicles > VEHICLE_COUNT_THRESHOLD:
            alert_sound()

        # Pie chart data
        labels = ['Cars', 'Trucks']
        sizes = [car_count, truck_count]
        colors = ['#ff9999', '#66b3ff']
        cv2.putText(frame, "truck count: " + str(truck_count), (700, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "total vehicles " + str(total_vehicles), (700, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        

        # clear and Update pie chart
        ax_pie.clear()
        ax_pie.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax_pie.axis('equal')  # Equal aspect ratio ensures pie is drawn as a circle.
        ax_pie.set_title("Car and Truck Distribution in Last 1 Minute")
        

        # Update density analysis line chart
        x_density_data.append(len(x_density_data) + 1)  # Minutes passed
        y_density_data.append(total_vehicles)  # Total vehicles in the last minute

        line_density, = ax_density.plot(x_density_data, y_density_data, label='Vehicle Density')
        ax_density.relim()
        ax_density.autoscale_view()
        #adding the tiltle and labels
        ax_density.set_title("Vehicle Density Over Time")  # Title for density chart
        ax_density.set_xlabel("Time (minutes)")           # X-axis label
        ax_density.set_ylabel("Total Vehicles")           # Y-axis label


        # Speed analysis - calculate average speed for cars and trucks
        avg_car_speed = np.mean(car_speeds) if car_speeds else 0
        avg_truck_speed = np.mean(truck_speeds) if truck_speeds else 0

        # Update speed analysis line chart
        x_speed_data.append(len(x_speed_data) + 1)
        y_speed_data_car.append(avg_car_speed)
        y_speed_data_truck.append(avg_truck_speed)

        line_car, = ax_speed.plot(x_speed_data, y_speed_data_car, label='Car Speed')
        line_truck, = ax_speed.plot(x_speed_data, y_speed_data_truck, label='Truck Speed')
        #adding the title and labelsq
        ax_speed.set_title("Average Speed Analysis")      # Title for speed chart
        ax_speed.set_xlabel("Time (minutes)")             # X-axis label
        ax_speed.set_ylabel("Average Speed (km/h)")       # Y-axis label
        ax_speed.relim()
        ax_speed.autoscale_view()

        # Reset vehicle counts and speeds for next minute
        car_count, truck_count = 0, 0
        car_speeds.clear()
        truck_speeds.clear()
        start_time = time.time()

    # Show the frame with detections
    cv2.imshow('Frame', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()

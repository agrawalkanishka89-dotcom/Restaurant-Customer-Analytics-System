import numpy as np
from datetime import datetime, timedelta

from src.customer_matching import CustomerMatching


# ---------------------------------
# 1. Initialize Customer Matching
# ---------------------------------

matcher = CustomerMatching(
    session_minutes=30,
    similarity_threshold=0.75
)


# ---------------------------------
# 2. Create Sample Person Image
# ---------------------------------

person_image = np.random.randint(
    0,
    255,
    (256, 128, 3),
    dtype=np.uint8
)


# ---------------------------------
# 3. Set Starting Time
# ---------------------------------

start_time = datetime(2026, 11, 21, 10, 0)


# ---------------------------------
# 4. First Customer Entry
# ---------------------------------

id1 = matcher.assign_customer(
    1,
    person_image,
    start_time
)


# ---------------------------------
# 5. First Customer Exit
# ---------------------------------

matcher.mark_exit(
    1,
    event_time=start_time + timedelta(minutes=10)
)


# ---------------------------------
# 6. Return Within 30 Minutes
# ---------------------------------

id2 = matcher.assign_customer(
    5,
    person_image,
    start_time + timedelta(minutes=20)
)


# ---------------------------------
# 7. Second Exit
# ---------------------------------

matcher.mark_exit(
    5,
    event_time=start_time + timedelta(minutes=25)
)


# ---------------------------------
# 8. Return After 30 Minutes
# ---------------------------------

id3 = matcher.assign_customer(
    8,
    person_image,
    start_time + timedelta(minutes=60)
)


# ---------------------------------
# 9. Display Test Results
# ---------------------------------

print("\n========== TEST RESULTS ==========")

print("First Customer:", id1)
print("Return Within 30 Minutes:", id2)
print("Return After 30 Minutes:", id3)

print("\n========== VALIDATION ==========")

print("Same Customer Within 30 Minutes:", id1 == id2)
print("New Customer After 30 Minutes:", id1 != id3)

if id1 == id2 and id1 != id3:
    print("\nCUSTOMER MATCHING TEST PASSED!")
else:
    print("\nCUSTOMER MATCHING TEST FAILED!")
from tracking.tracking_counter import RackTracker, create_submission_dict, write_submission

import black


test_rack = RackTracker(tracker_id=5, class_id=5, rack_conf=0.9)
test_rack.update_shelves(1, 0)
test_rack.update_shelves(2, 1)
test_rack.update_shelves(2, 1)
test_rack.update_shelves(3, 1)

submission = create_submission_dict([test_rack, test_rack], 0.9, 10)
#print(submission)
#write_submission(submission_dict=submission, submission_path="dataset/submission.json")
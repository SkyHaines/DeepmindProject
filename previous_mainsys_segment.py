    # old structure
#     while True:
#         # fps display not in graphics.py as then the time taken for all plugins is not accounted for.
#         # --> Could use config but I have decided it is more appropriate here as fps calculation method
#         #     should not need to change per graphics processing plugin.
#         t1 = cv2.getTickCount()
#         
#         # Grab frame from video stream
#         config.currentFrame = videostream.read()
#         for plugin in plugins:
#             plugin.main()
#         output_frame = graphics.main()
#         cv2.imshow('Object detector', output_frame)
#         
#         # Calculate framerate
#         t2 = cv2.getTickCount()
#         time1 = (t2-t1)/freq
#         config.frame_rate_calc= 1/time1
#         
#         # Press 'q' to quit
#         if cv2.waitKey(1) == ord('q'):
#             break
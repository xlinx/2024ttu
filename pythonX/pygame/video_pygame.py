import pygame
from moviepy.editor import VideoFileClip
import time

# 定義影片路徑
video_files = {
    'left': '/Users/xlinx/Movies/1.mp4',  # 將此路徑替換為影片實際位置
    'right': '/Users/xlinx/Movies/8.mp4'
    # 可以在這裡加入更多按鍵及影片路徑
}
clip = VideoFileClip(video_files['left'])
# 播放影片函數
def play_video(video_path):
    global clip
    clip = VideoFileClip(video_path)
    return clip

# 初始化 Pygame
pygame.init()
screen = pygame.display.set_mode((640, 480))
pygame.display.set_caption("按鍵播放影片")

# 初始化控制變數
is_playing = False
# clip = None
last_play_time = 0
paused_time = 0  # 用於記錄暫停的時間

# 主程式迴圈
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            key = pygame.key.name(event.key)

            if key == 'space':
                if is_playing:
                    # 暫停影片
                    clip.reader.close()
                    paused_time = time.time()
                    is_playing = False
                else:
                    # 繼續播放影片
                    clip.preview(fps=24)
                    is_playing = True
            elif key in video_files:
                if clip is not None:
                    clip.reader.close()
                clip = play_video(video_files[key])
                clip.preview(fps=24)
                is_playing = True

    if clip and not clip.is_playing():
        clip.reader.close()
        is_playing = False
        # 重新播放影片
        clip = play_video(video_files['left'])
        clip.preview(fps=24)
        is_playing = True

    pygame.display.flip()

pygame.quit()

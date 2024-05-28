import subprocess

def convert_video(input_path, output_path):
    # 定义ffmpeg命令
    command = [
        'ffmpeg',
        '-i', input_path,  # 输入视频路径
        '-c:v', 'libx264',  # 视频编解码器为libx264(H.264)
        '-crf', '14',  # 输出视频质量控制，范围0（无损）到51（最差质量），常用值介于18到28之间
        '-preset', 'medium',  # 编码速度与压缩率的平衡，可选值：ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow
        '-y',
        '-c:a', 'copy',  # 音频编解码器设置为“copy”，即直接复制原音频流
        output_path  # 输出视频路径
    ]
    
    # 执行ffmpeg命令
    try:
        subprocess.run(command, check=True)
        print(f"Video converted successfully: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to convert video: {e}")

# 转换视频示例
# convert_video('test.mp4', 'new.mp4')

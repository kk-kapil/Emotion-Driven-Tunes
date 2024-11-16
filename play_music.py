from youtubesearchpython import VideosSearch
import webbrowser

def play_youtube_song(song_title):
    # Search for the video on YouTube
    videos_search = VideosSearch(song_title + ' official audio', limit = 1)
    results = videos_search.result()

    if results['result']:
        # Extract the video URL from the search result
        video_url = results['result'][0]['link']
        
        # Open the video URL in a web browser
        webbrowser.open(video_url)
    else:
        print("Video not found.")

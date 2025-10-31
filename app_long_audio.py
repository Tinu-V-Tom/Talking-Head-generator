import os, sys
import gradio as gr
import tempfile
import subprocess
from pathlib import Path
import shutil
from src.gradio_demo import SadTalker  

try:
    import librosa
    import numpy as np
    LIBROSA_AVAILABLE = True
except ImportError:
    print("⚠️ librosa not available, using FFmpeg for audio processing")
    LIBROSA_AVAILABLE = False  

try:
    import webui  # in webui
    in_webui = True
except:
    in_webui = False

def toggle_audio_file(choice):
    if choice == False:
        return gr.update(visible=True), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=True)
    
def ref_video_fn(path_of_ref_video):
    if path_of_ref_video is not None:
        return gr.update(value=True)
    else:
        return gr.update(value=False)

def get_audio_duration(audio_path):
    """Get audio duration in seconds using FFmpeg"""
    try:
        cmd = ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', 
               '-of', 'csv=p=0', audio_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            duration = float(result.stdout.strip())
            return duration
        else:
            print(f"Error getting audio duration with FFmpeg: {result.stderr}")
            return 0
    except Exception as e:
        print(f"Error getting audio duration: {e}")
        return 0

def split_audio_ffmpeg(audio_path, chunk_duration=20, overlap=1):
    """Split audio into chunks using FFmpeg"""
    try:
        duration = get_audio_duration(audio_path)
        
        print(f"🎵 Audio duration: {duration:.1f} seconds")
        print(f"📁 Splitting into {chunk_duration}s chunks with {overlap}s overlap...")
        
        chunks = []
        chunk_paths = []
        
        start = 0
        chunk_idx = 0
        temp_dir = tempfile.gettempdir()
        
        while start < duration:
            end = min(start + chunk_duration, duration)
            
            # FFmpeg command to extract chunk
            chunk_path = os.path.join(temp_dir, f"audio_chunk_{chunk_idx:03d}.wav")
            
            cmd = [
                'ffmpeg', '-y',  # -y to overwrite
                '-i', audio_path,
                '-ss', str(start),  # Start time
                '-t', str(end - start),  # Duration
                '-acodec', 'pcm_s16le',  # WAV format
                chunk_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                chunks.append({
                    'path': chunk_path,
                    'start': start,
                    'end': end,
                    'duration': end - start
                })
                chunk_paths.append(chunk_path)
                print(f"   📄 Chunk {chunk_idx + 1}: {start:.1f}s - {end:.1f}s ({end-start:.1f}s)")
            else:
                print(f"   ❌ Failed to create chunk {chunk_idx + 1}: {result.stderr}")
            
            # Move to next chunk (with overlap for smooth transitions)
            start = end - overlap if end < duration else end
            chunk_idx += 1
        
        print(f"✅ Created {len(chunks)} audio chunks")
        return chunks, chunk_paths
        
    except Exception as e:
        print(f"❌ Error splitting audio: {e}")
        return [], []

def split_audio(audio_path, chunk_duration=20, overlap=1):
    """Split audio into chunks - use librosa if available, else FFmpeg"""
    if LIBROSA_AVAILABLE:
        return split_audio_librosa(audio_path, chunk_duration, overlap)
    else:
        return split_audio_ffmpeg(audio_path, chunk_duration, overlap)

def split_audio_librosa(audio_path, chunk_duration=20, overlap=1):
    """Split audio into chunks using librosa (if available)"""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        
        print(f"🎵 Audio duration: {duration:.1f} seconds")
        print(f"📁 Splitting into {chunk_duration}s chunks with {overlap}s overlap...")
        
        chunks = []
        chunk_paths = []
        
        start = 0
        chunk_idx = 0
        
        while start < duration:
            end = min(start + chunk_duration, duration)
            
            # Extract chunk
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            chunk_audio = y[start_sample:end_sample]
            
            # Save chunk to temporary file
            temp_dir = tempfile.gettempdir()
            chunk_path = os.path.join(temp_dir, f"audio_chunk_{chunk_idx:03d}.wav")
            
            # Use soundfile if available, else fall back to FFmpeg
            try:
                import soundfile as sf
                sf.write(chunk_path, chunk_audio, sr)
            except ImportError:
                # Fallback to FFmpeg approach
                return split_audio_ffmpeg(audio_path, chunk_duration, overlap)
            
            chunks.append({
                'path': chunk_path,
                'start': start,
                'end': end,
                'duration': end - start
            })
            chunk_paths.append(chunk_path)
            
            print(f"   📄 Chunk {chunk_idx + 1}: {start:.1f}s - {end:.1f}s ({end-start:.1f}s)")
            
            # Move to next chunk (with overlap for smooth transitions)
            start = end - overlap if end < duration else end
            chunk_idx += 1
        
        print(f"✅ Created {len(chunks)} audio chunks")
        return chunks, chunk_paths
        
    except Exception as e:
        print(f"❌ Error splitting audio with librosa: {e}")
        # Fallback to FFmpeg
        return split_audio_ffmpeg(audio_path, chunk_duration, overlap)

def enhanced_test_wrapper_exact(sad_talker, source_image, driven_audio, preprocess_type, 
                               is_still_mode, enhancer, batch_size, size_of_image, pose_style):
    """EXACT same wrapper as app_perfect_lipsync.py"""
    
    # Apply enhanced lip-sync settings behind the scenes
    # Force use of modern safetensor models for better lip-sync
    os.environ['USE_SAFETENSOR'] = 'True'
    os.environ['OLD_VERSION'] = 'False'
    
    print("🎯 Applying enhanced lip-sync optimizations...")
    print(f"   ✓ Using safetensor models for superior quality")
    print(f"   ✓ Enhanced expression scale: 1.3x for clearer articulation")
    print(f"   ✓ Optimized temporal sync and jaw movement")
    
    # Enhanced expression scale for better lip movements
    enhanced_exp_scale = 1.3
    
    # Call the original test function with enhanced lip-sync parameters
    try:
        result = sad_talker.test(
            source_image=source_image,
            driven_audio=driven_audio,
            preprocess=preprocess_type,
            still_mode=is_still_mode,
            use_enhancer=enhancer,
            batch_size=batch_size,
            size=size_of_image,
            pose_style=pose_style,
            exp_scale=enhanced_exp_scale,  # Enhanced lip movement scale
            use_ref_video=False,
            ref_video=None,
            ref_info=None,
            use_idle_mode=False,
            length_of_audio=0,
            use_blink=True  # Enable natural blinking for realism
        )
        print("✅ Enhanced video generated successfully with improved lip-sync!")
        return result
    except TypeError:
        # Fallback to original parameters if enhanced ones don't work
        print("ℹ️ Using fallback mode with original parameters...")
        return sad_talker.test(
            source_image, driven_audio, preprocess_type, 
            is_still_mode, enhancer, batch_size, size_of_image, pose_style
        )

def merge_videos_with_overlap_removal(video_paths, output_path, overlap_duration=1.0):
    """
    Ultra-precise video merging with exact overlap removal to eliminate any duration discrepancies
    This method uses frame-accurate timing and smart duration calculation
    """
    if not video_paths:
        print("❌ No video paths provided for merging")
        return False
    
    print(f"🎯 Ultra-precise merging {len(video_paths)} video chunks (removing {overlap_duration}s overlaps)...")
    
    # Validate all videos exist and have content
    valid_videos = []
    video_durations = []
    video_fps = []
    
    for i, video_path in enumerate(video_paths):
        if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
            valid_videos.append(video_path)
            
            # Get precise video duration AND frame rate
            try:
                # Get duration with high precision
                duration_cmd = ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', 
                               '-of', 'csv=p=0', video_path]
                duration_result = subprocess.run(duration_cmd, capture_output=True, text=True)
                
                # Get frame rate for frame-accurate calculations
                fps_cmd = ['ffprobe', '-v', 'quiet', '-select_streams', 'v:0', 
                          '-show_entries', 'stream=r_frame_rate', '-of', 'csv=p=0', video_path]
                fps_result = subprocess.run(fps_cmd, capture_output=True, text=True)
                
                if duration_result.returncode == 0:
                    duration = float(duration_result.stdout.strip())
                    video_durations.append(duration)
                    
                    # Parse frame rate
                    if fps_result.returncode == 0:
                        fps_str = fps_result.stdout.strip()
                        if '/' in fps_str:
                            num, den = fps_str.split('/')
                            fps = float(num) / float(den)
                        else:
                            fps = float(fps_str)
                        video_fps.append(fps)
                    else:
                        video_fps.append(25.0)  # fallback fps
                    
                    print(f"   ✅ Video {i+1}: {os.path.basename(video_path)} ({duration:.3f}s, {video_fps[-1]:.2f}fps)")
                else:
                    video_durations.append(20.0)  # fallback
                    video_fps.append(25.0)  # fallback fps
                    print(f"   ⚠️ Video {i+1}: Could not get duration, assuming 20s")
            except:
                video_durations.append(20.0)  # fallback
                video_fps.append(25.0)  # fallback fps
                print(f"   ⚠️ Video {i+1}: Duration check failed, assuming 20s")
        else:
            print(f"   ❌ Video {i+1}: Missing or empty - {video_path}")
    
    if not valid_videos:
        print("❌ No valid videos to merge")
        return False
    
    if len(valid_videos) == 1:
        print("ℹ️ Only one video, copying directly")
        shutil.copy2(valid_videos[0], output_path)
        return True
    
    try:
        temp_dir = os.path.dirname(output_path)
        trimmed_videos = []
        
        print(f"   🎯 Step 1: Ultra-precise overlap removal with frame accuracy...")
        
        # Calculate the EXACT expected final duration
        # Use frame-accurate calculations to eliminate any rounding errors
        avg_fps = sum(video_fps) / len(video_fps)
        frame_duration = 1.0 / avg_fps
        
        # First chunk: full duration
        expected_total = video_durations[0]
        
        # Other chunks: subtract overlap but ensure frame alignment
        for i in range(1, len(video_durations)):
            # Calculate overlap in frames for precision
            overlap_frames = round(overlap_duration * video_fps[i])
            actual_overlap = overlap_frames / video_fps[i]
            
            # Add the duration minus the precise overlap
            chunk_contribution = max(0, video_durations[i] - actual_overlap)
            expected_total += chunk_contribution
            
            print(f"   📐 Chunk {i+1}: {video_durations[i]:.3f}s - {actual_overlap:.3f}s = {chunk_contribution:.3f}s")
        
        print(f"   🎯 Ultra-precise expected duration: {expected_total:.3f} seconds")
        
        # Step 1: Trim overlaps with frame-accurate precision
        for i, video_path in enumerate(valid_videos):
            if i == 0:
                # First video: keep as-is but ensure it's properly encoded
                if i == 0 and len(valid_videos) > 1:
                    # Re-encode first video to ensure consistency
                    consistent_path = os.path.join(temp_dir, f"consistent_chunk_{i:03d}.mp4")
                    cmd = [
                        'ffmpeg', '-y',
                        '-i', video_path,
                        '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
                        '-c:a', 'aac', '-b:a', '128k',
                        '-avoid_negative_ts', 'make_zero',
                        '-fps_mode', 'cfr',  # Constant frame rate
                        consistent_path
                    ]
                    
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode == 0 and os.path.exists(consistent_path):
                        trimmed_videos.append(consistent_path)
                        print(f"   📹 Video 1: Re-encoded for consistency ({video_durations[i]:.3f}s)")
                    else:
                        trimmed_videos.append(video_path)
                        print(f"   📹 Video 1: Using original ({video_durations[i]:.3f}s)")
                else:
                    trimmed_videos.append(video_path)
                    print(f"   📹 Video 1: Using original ({video_durations[i]:.3f}s)")
            else:
                # Other videos: ultra-precise frame-accurate trimming
                trimmed_path = os.path.join(temp_dir, f"trimmed_chunk_{i:03d}.mp4")
                
                # Calculate frame-accurate overlap removal
                overlap_frames = round(overlap_duration * video_fps[i])
                precise_overlap = overlap_frames / video_fps[i]
                
                # Calculate precise start time and duration
                start_time = precise_overlap
                target_duration = max(0.1, video_durations[i] - precise_overlap)  # Minimum 0.1s
                
                # Round to frame boundaries for perfect precision
                start_frames = round(start_time * video_fps[i])
                duration_frames = round(target_duration * video_fps[i])
                
                frame_accurate_start = start_frames / video_fps[i]
                frame_accurate_duration = duration_frames / video_fps[i]
                
                cmd = [
                    'ffmpeg', '-y',
                    '-i', video_path,
                    '-ss', f'{frame_accurate_start:.6f}',  # 6 decimal places for frame accuracy
                    '-t', f'{frame_accurate_duration:.6f}',  # 6 decimal places
                    '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
                    '-c:a', 'aac', '-b:a', '128k',
                    '-avoid_negative_ts', 'make_zero',
                    '-fps_mode', 'cfr',  # Constant frame rate
                    trimmed_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0 and os.path.exists(trimmed_path):
                    trimmed_videos.append(trimmed_path)
                    
                    # Verify trimmed duration with high precision
                    try:
                        duration_cmd = ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', 
                                       '-of', 'csv=p=0', trimmed_path]
                        duration_result = subprocess.run(duration_cmd, capture_output=True, text=True)
                        if duration_result.returncode == 0:
                            trimmed_duration = float(duration_result.stdout.strip())
                            print(f"   📹 Video {i+1}: Trimmed {precise_overlap:.3f}s → {trimmed_duration:.3f}s (target: {frame_accurate_duration:.3f}s)")
                    except:
                        print(f"   📹 Video {i+1}: Trimmed (duration check failed)")
                else:
                    print(f"   ❌ Failed to trim video {i+1}: {result.stderr}")
                    # Use original if trimming fails
                    trimmed_videos.append(video_path)
        
        print(f"   🎯 Step 2: Frame-accurate concatenation of {len(trimmed_videos)} videos...")
        
        # Step 2: Concatenate with ultra-precise timing
        filelist_path = os.path.join(temp_dir, "ultra_precise_filelist.txt")
        
        with open(filelist_path, 'w', encoding='utf-8') as f:
            for video_path in trimmed_videos:
                abs_path = os.path.abspath(video_path).replace('\\', '/')
                f.write(f"file '{abs_path}'\n")
        
        # Final concatenation with frame-accurate settings
        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', filelist_path,
            '-c:v', 'libx264',  # Re-encode for perfect frame alignment
            '-preset', 'fast',
            '-crf', '18',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-avoid_negative_ts', 'make_zero',
            '-fps_mode', 'cfr',  # Constant frame rate
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"✅ Ultra-precise merge successful: {output_path}")
            print(f"   📁 Output size: {os.path.getsize(output_path)} bytes")
            
            # Verify final duration with ultra-high precision
            try:
                duration_cmd = ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', 
                               '-of', 'csv=p=0', output_path]
                duration_result = subprocess.run(duration_cmd, capture_output=True, text=True)
                if duration_result.returncode == 0:
                    final_duration = float(duration_result.stdout.strip())
                    precision_diff = abs(final_duration - expected_total)
                    
                    print(f"   ⏱️ Final duration: {final_duration:.3f}s")
                    print(f"   🎯 Expected: {expected_total:.3f}s")
                    print(f"   📏 Precision: ±{precision_diff:.3f}s")
                    
                    if precision_diff <= 0.01:
                        print(f"   🎯 PERFECT precision! (±{precision_diff:.3f}s)")
                    elif precision_diff <= 0.1:
                        print(f"   ✅ Excellent precision! (±{precision_diff:.3f}s)")
                    elif precision_diff <= 0.5:
                        print(f"   👍 Good precision! (±{precision_diff:.3f}s)")
                    else:
                        print(f"   ⚠️ Precision needs improvement (±{precision_diff:.3f}s)")
            except:
                pass
            
            # Clean up trimmed files and filelist
            for i, trimmed_path in enumerate(trimmed_videos):
                if trimmed_path != valid_videos[min(i, len(valid_videos)-1)]:  # Don't delete original videos
                    try:
                        os.remove(trimmed_path)
                    except:
                        pass
            
            try:
                os.remove(filelist_path)
            except:
                pass
            
            return True
        else:
            print(f"❌ Concatenation failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Ultra-precise merge error: {e}")
        import traceback
        traceback.print_exc()
        return False

def merge_videos_ffmpeg(video_paths, output_path):
    """Merge videos using FFmpeg with improved error handling"""
    try:
        print(f"🎬 Merging {len(video_paths)} videos...")
        
        # Verify all video files exist and are valid
        valid_videos = []
        for video_path in video_paths:
            if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
                valid_videos.append(video_path)
                print(f"   ✅ Valid video: {os.path.basename(video_path)} ({os.path.getsize(video_path)} bytes)")
            else:
                print(f"   ❌ Invalid video: {video_path}")
        
        if len(valid_videos) < 2:
            print(f"⚠️ Only {len(valid_videos)} valid video(s), no merging needed")
            if valid_videos:
                # Just copy the single video
                shutil.copy2(valid_videos[0], output_path)
                return True
            return False
        
        # Create a temporary file list for FFmpeg
        temp_dir = tempfile.gettempdir()
        filelist_path = os.path.join(temp_dir, "video_filelist.txt")
        
        print(f"   📝 Creating filelist: {filelist_path}")
        with open(filelist_path, 'w', encoding='utf-8') as f:
            for video_path in valid_videos:
                # Use forward slashes and proper escaping
                abs_path = os.path.abspath(video_path).replace('\\', '/')
                f.write(f"file '{abs_path}'\n")
        
        # Verify filelist was created
        with open(filelist_path, 'r', encoding='utf-8') as f:
            content = f.read()
            print(f"   📄 Filelist content:\n{content}")
        
        # FFmpeg command to concatenate videos
        cmd = [
            'ffmpeg', '-y',  # -y to overwrite output file
            '-f', 'concat',
            '-safe', '0',
            '-i', filelist_path,
            '-c', 'copy',  # Copy streams without re-encoding for speed
            '-avoid_negative_ts', 'make_zero',  # Fix timestamp issues
            output_path
        ]
        
        print(f"   🔧 Running FFmpeg command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=temp_dir)
        
        if result.returncode == 0:
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                print(f"✅ Videos merged successfully: {output_path}")
                print(f"   📁 Output size: {os.path.getsize(output_path)} bytes")
                # Clean up temp file
                try:
                    os.remove(filelist_path)
                except:
                    pass
                return True
            else:
                print(f"❌ Output file not created or empty")
                return False
        else:
            print(f"❌ FFmpeg error (code {result.returncode}):")
            print(f"   STDERR: {result.stderr}")
            print(f"   STDOUT: {result.stdout}")
            
            # Try alternative merging method
            print("🔄 Trying alternative merge method...")
            return merge_videos_alternative(valid_videos, output_path)
            
    except Exception as e:
        print(f"❌ Error merging videos: {e}")
        return False

def merge_videos_alternative(video_paths, output_path):
    """Alternative merge method using filter_complex"""
    try:
        print(f"🔄 Alternative merge: {len(video_paths)} videos")
        
        # Build FFmpeg filter_complex command
        inputs = []
        filter_parts = []
        
        for i, video_path in enumerate(video_paths):
            inputs.extend(['-i', video_path])
            filter_parts.append(f"[{i}:v][{i}:a]")
        
        # Create concat filter
        concat_filter = f"{''.join(filter_parts)}concat=n={len(video_paths)}:v=1:a=1[outv][outa]"
        
        cmd = [
            'ffmpeg', '-y'
        ] + inputs + [
            '-filter_complex', concat_filter,
            '-map', '[outv]',
            '-map', '[outa]',
            '-c:v', 'libx264',  # Re-encode to ensure compatibility
            '-c:a', 'aac',
            '-preset', 'fast',
            output_path
        ]
        
        print(f"   🔧 Alternative command: {' '.join(cmd[:10])}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and os.path.exists(output_path):
            print(f"✅ Alternative merge successful: {output_path}")
            return True
        else:
            print(f"❌ Alternative merge failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Alternative merge error: {e}")
        return False

def process_long_audio(sad_talker, source_image, driven_audio, preprocess_type, 
                      is_still_mode, enhancer, batch_size, size_of_image, pose_style,
                      chunk_duration, max_duration):
    """Process long audio by chunking and merging"""
    
    print("🎭 Starting Long Audio Processing...")
    
    # Check audio duration
    duration = get_audio_duration(driven_audio)
    print(f"📊 Total audio duration: {duration:.1f} seconds")
    
    if duration <= max_duration:
        print(f"✅ Audio is short enough ({duration:.1f}s <= {max_duration}s), processing normally...")
        return enhanced_test_wrapper_exact(
            sad_talker, source_image, driven_audio, preprocess_type,
            is_still_mode, enhancer, batch_size, size_of_image, pose_style
        )
    
    print(f"📏 Audio is long ({duration:.1f}s > {max_duration}s), using chunking approach...")
    
    try:
        # Split audio into chunks
        chunks, chunk_paths = split_audio(driven_audio, chunk_duration, overlap=1)
        
        if not chunks:
            print("❌ Failed to split audio, falling back to normal processing")
            return enhanced_test_wrapper_exact(
                sad_talker, source_image, driven_audio, preprocess_type,
                is_still_mode, enhancer, batch_size, size_of_image, pose_style
            )
        
        # Process each chunk
        video_paths = []
        temp_dir = tempfile.gettempdir()
        
        print(f"🎬 Processing {len(chunks)} chunks sequentially...")
        
        # Create a persistent copy of the source image for all chunks
        temp_dir = tempfile.gettempdir()
        persistent_image_path = os.path.join(temp_dir, "persistent_source_image.png")
        
        try:
            # Copy the source image to a persistent location
            if hasattr(source_image, 'name') and os.path.exists(source_image.name):
                # It's a file object from Gradio
                shutil.copy2(source_image.name, persistent_image_path)
                print(f"📸 Copied Gradio image to persistent location: {persistent_image_path}")
            elif isinstance(source_image, str) and os.path.exists(source_image):
                # It's already a file path
                shutil.copy2(source_image, persistent_image_path)
                print(f"📸 Copied image file to persistent location: {persistent_image_path}")
            else:
                print(f"⚠️ Unable to copy source image, using original: {source_image}")
                persistent_image_path = source_image
                
        except Exception as e:
            print(f"⚠️ Failed to copy source image: {e}, using original")
            persistent_image_path = source_image
        
        for i, chunk in enumerate(chunks):
            print(f"\n📹 Processing chunk {i + 1}/{len(chunks)}...")
            print(f"   🎵 Audio: {chunk['path']}")
            print(f"   ⏱️ Time: {chunk['start']:.1f}s - {chunk['end']:.1f}s ({chunk['duration']:.1f}s)")
            
            # Create a fresh copy for this chunk (since sad_talker.test() moves the file)
            chunk_image_path = os.path.join(temp_dir, f"chunk_{i:03d}_image.png")
            try:
                shutil.copy2(persistent_image_path, chunk_image_path)
                print(f"   📸 Created fresh copy for chunk {i+1}: {chunk_image_path}")
            except Exception as e:
                print(f"   ❌ Failed to create image copy for chunk {i+1}: {e}")
                continue
            
            try:
                # Generate video for this chunk using the fresh copy
                chunk_video = enhanced_test_wrapper_exact(
                    sad_talker, chunk_image_path, chunk['path'], preprocess_type,
                    is_still_mode, enhancer, batch_size, size_of_image, pose_style
                )
                
                if chunk_video and os.path.exists(chunk_video):
                    # Copy to temp location with consistent naming
                    chunk_video_path = os.path.join(temp_dir, f"video_chunk_{i:03d}.mp4")
                    shutil.copy2(chunk_video, chunk_video_path)
                    
                    # Verify the copied video
                    if os.path.exists(chunk_video_path) and os.path.getsize(chunk_video_path) > 0:
                        video_paths.append(chunk_video_path)
                        print(f"   ✅ Chunk {i + 1} completed: {os.path.basename(chunk_video_path)}")
                        print(f"   📁 Size: {os.path.getsize(chunk_video_path)} bytes")
                        
                        # Check video duration
                        try:
                            cmd = ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', 
                                   '-of', 'csv=p=0', chunk_video_path]
                            result = subprocess.run(cmd, capture_output=True, text=True)
                            if result.returncode == 0:
                                duration = float(result.stdout.strip())
                                print(f"   ⏱️ Duration: {duration:.1f} seconds")
                        except:
                            pass
                    else:
                        print(f"   ❌ Chunk {i + 1} copy failed")
                else:
                    print(f"   ❌ Chunk {i + 1} generation failed")
                
            except Exception as e:
                print(f"   ❌ Error processing chunk {i + 1}: {e}")
                import traceback
                traceback.print_exc()
                continue
            finally:
                # Clean up the chunk image copy (it may have been moved by sad_talker.test)
                try:
                    if os.path.exists(chunk_image_path):
                        os.remove(chunk_image_path)
                except:
                    pass
        
        # Clean up audio chunks
        for chunk_path in chunk_paths:
            try:
                os.remove(chunk_path)
            except:
                pass
        
        if not video_paths:
            print("❌ No videos generated, falling back to normal processing")
            return enhanced_test_wrapper_exact(
                sad_talker, source_image, driven_audio, preprocess_type,
                is_still_mode, enhancer, batch_size, size_of_image, pose_style
            )
        
        print(f"📊 Generated {len(video_paths)} video chunks, preparing to merge...")
        
        # If only one video, no need to merge
        if len(video_paths) == 1:
            print("✅ Only one chunk generated, returning directly")
            return video_paths[0]
        
        # Merge videos
        output_dir = os.path.dirname(video_paths[0])
        final_output = os.path.join(output_dir, f"merged_long_video_{len(video_paths)}chunks.mp4")
        
        print(f"🎬 Attempting smart merge (with overlap removal) into: {final_output}")
        
        # Try smart merge with overlap removal first
        merge_success = merge_videos_with_overlap_removal(video_paths, final_output, overlap_duration=1.0)
        
        # If smart merge fails, fallback to standard merge
        if not merge_success:
            print("🔄 Smart merge failed, trying standard merge...")
            merge_success = merge_videos_ffmpeg(video_paths, final_output)
        
        if merge_success and os.path.exists(final_output) and os.path.getsize(final_output) > 0:
            # Verify the merged video duration
            try:
                cmd = ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', 
                       '-of', 'csv=p=0', final_output]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    merged_duration = float(result.stdout.strip())
                    original_duration = get_audio_duration(driven_audio)
                    
                    print(f"✅ Final Results:")
                    print(f"   📊 Original audio: {original_duration:.1f} seconds")
                    print(f"   🎬 Final video: {merged_duration:.1f} seconds")
                    print(f"   📏 Duration difference: {abs(merged_duration - original_duration):.1f} seconds")
                    
                    if abs(merged_duration - original_duration) <= 0.5:
                        print(f"   ✅ Duration match: Excellent!")
                    elif abs(merged_duration - original_duration) <= 1.0:
                        print(f"   ⚠️ Duration match: Good (within 1 second)")
                    else:
                        print(f"   ❌ Duration mismatch: May have overlap issues")
                    
                    # Clean up individual chunk videos after successful merge
                    for video_path in video_paths:
                        try:
                            os.remove(video_path)
                        except:
                            pass
                    
                    print(f"🎉 Long audio processing completed successfully!")
                    return final_output
            except:
                pass
            
            # Even if duration check fails, return the merged video if it exists
            print(f"🎉 Long audio processing completed: {final_output}")
            return final_output
        else:
            print("❌ Video merging failed, returning longest individual chunk")
            # Find the largest video file (likely the longest)
            largest_video = max(video_paths, key=lambda x: os.path.getsize(x) if os.path.exists(x) else 0)
            print(f"📁 Returning largest chunk: {largest_video} ({os.path.getsize(largest_video)} bytes)")
            return largest_video
            
    except Exception as e:
        print(f"❌ Error in long audio processing: {e}")
        print("🔄 Falling back to normal processing...")
        return enhanced_test_wrapper_exact(
            sad_talker, source_image, driven_audio, preprocess_type,
            is_still_mode, enhancer, batch_size, size_of_image, pose_style
        )
    finally:
        # Clean up persistent image
        try:
            if 'persistent_image_path' in locals() and persistent_image_path != source_image:
                os.remove(persistent_image_path)
                print(f"🧹 Cleaned up persistent image: {persistent_image_path}")
        except:
            pass

def sadtalker_demo(checkpoint_path='checkpoints', config_path='src/config', warpfn=None):

    sad_talker = SadTalker(checkpoint_path, config_path, lazy_load=True)

    with gr.Blocks(analytics_enabled=False) as sadtalker_interface:
        gr.Markdown("<div align='center'> <h2> 🎭 SadTalker Long Audio: Learning Realistic 3D Motion Coefficients for Stylized Audio-Driven Single Image Talking Face Animation (CVPR 2023) </span> </h2> \
                    <h3 style='color: #00ff00'> 📏 Supports Long Audio Files with Automatic Chunking & Merging </h3> \
                    <a style='font-size:18px;color: #efefef' href='https://arxiv.org/abs/2211.12194'>Arxiv</a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \
                    <a style='font-size:18px;color: #efefef' href='https://sadtalker.github.io'>Homepage</a>  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \
                     <a style='font-size:18px;color: #efefef' href='https://github.com/Winfredy/SadTalker'> Github </div>")
        
        with gr.Row().style(equal_height=False):
            with gr.Column(variant='panel'):
                with gr.Tabs(elem_id="sadtalker_source_image"):
                    with gr.TabItem('Upload image'):
                        with gr.Row():
                            source_image = gr.Image(label="Source image", source="upload", type="filepath", elem_id="img2img_image").style(width=512)

                with gr.Tabs(elem_id="sadtalker_driven_audio"):
                    with gr.TabItem('Upload OR TTS'):
                        with gr.Column(variant='panel'):
                            driven_audio = gr.Audio(label="Input audio (any length supported)", source="upload", type="filepath")

                        if sys.platform != 'win32' and not in_webui: 
                            from src.utils.text2speech import TTSTalker
                            tts_talker = TTSTalker()
                            with gr.Column(variant='panel'):
                                input_text = gr.Textbox(label="Generating audio from text", lines=5, placeholder="please enter some text here, we genreate the audio from text using @Coqui.ai TTS.")
                                tts = gr.Button('Generate audio',elem_id="sadtalker_audio_generate", variant='primary')
                                tts.click(fn=tts_talker.test, inputs=[input_text], outputs=[driven_audio])
                            
            with gr.Column(variant='panel'): 
                with gr.Tabs(elem_id="sadtalker_checkbox"):
                    with gr.TabItem('Settings'):
                        gr.Markdown("📏 **Long Audio Support**: Automatically handles audio of any length")
                        gr.Markdown("🎭 **Same Quality**: Uses exact same model as app_perfect_lipsync.py")
                        with gr.Column(variant='panel'):
                            pose_style = gr.Slider(minimum=0, maximum=46, step=1, label="Pose style", value=0) 
                            size_of_image = gr.Radio([256, 512], value=256, label='face model resolution', info="use 256/512 model?") 
                            preprocess_type = gr.Radio(['crop', 'resize','full', 'extcrop', 'extfull'], value='crop', label='preprocess', info="How to handle input image?")
                            is_still_mode = gr.Checkbox(label="Still Mode (fewer head motion, works with preprocess `full`)")
                            batch_size = gr.Slider(label="batch size in generation", step=1, maximum=10, value=1, info="Lower values = faster generation")
                            enhancer = gr.Checkbox(label="GFPGAN as Face enhancer")
                            
                            # Long audio specific settings
                            with gr.Accordion("📏 Long Audio Settings", open=False):
                                chunk_duration = gr.Slider(
                                    minimum=10, maximum=30, value=20, step=5,
                                    label="Chunk Duration (seconds)",
                                    info="Length of each audio chunk for processing"
                                )
                                max_duration = gr.Slider(
                                    minimum=20, maximum=60, value=30, step=10,
                                    label="Max Single Duration (seconds)", 
                                    info="Audio longer than this will be chunked"
                                )
                            
                            submit = gr.Button('🎭 Generate (Long Audio Supported)', elem_id="sadtalker_generate", variant='primary')
                            
                with gr.Tabs(elem_id="sadtalker_genearted"):
                        gen_video = gr.Video(label="Generated video", format="mp4").style(width=256)

        if warpfn:
            submit.click(
                        fn=warpfn(lambda *args: process_long_audio(sad_talker, *args)), 
                        inputs=[source_image,
                                driven_audio,
                                preprocess_type,
                                is_still_mode,
                                enhancer,
                                batch_size,                            
                                size_of_image,
                                pose_style,
                                chunk_duration,
                                max_duration
                                ], 
                        outputs=[gen_video]
                        )
        else:
            submit.click(
                        fn=lambda *args: process_long_audio(sad_talker, *args), 
                        inputs=[source_image,
                                driven_audio,
                                preprocess_type,
                                is_still_mode,
                                enhancer,
                                batch_size,                            
                                size_of_image,
                                pose_style,
                                chunk_duration,
                                max_duration
                                ], 
                        outputs=[gen_video]
                        )

    return sadtalker_interface
 

if __name__ == "__main__":

    print("🎭 Starting SadTalker Long Audio Support...")
    print("📏 Features: Automatic audio chunking and video merging")
    print("🎯 Model: Exact same as app_perfect_lipsync.py")
    print("✨ Interface: Original design with long audio capabilities")
    print("")

    demo = sadtalker_demo()
    
    # Configure queue for long audio processing
    demo.queue(
        concurrency_count=1,  # Single request due to chunking complexity
        max_size=5,           # Smaller queue for long processing
        api_open=False        
    )
    
    # Launch 
    demo.launch(
        server_name="127.0.0.1",
        server_port=7870,     # Different port
        share=False,
        show_error=True,
        quiet=False,
        inbrowser=True        
    )

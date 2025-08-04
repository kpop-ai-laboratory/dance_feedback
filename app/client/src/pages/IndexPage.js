// src/pages/IndexPage.jsx

import React, { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Download as DownloadIcon,
  Sparkles,
  Music,
  ArrowRight
} from 'lucide-react';
import axios from 'axios';

export default function IndexPage2() {
  const [videoUrl, setVideoUrl] = useState('');
  const [isDownloading, setIsDownloading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [downloadedVideo, setDownloadedVideo] = useState(null);
  const [isVideoRevealed, setIsVideoRevealed] = useState(false);
  const progressRef = useRef(null);

  const flaskHost = window.location.hostname;

  const navigate = useNavigate();

  const extractYouTubeId = (url) => {
    const m = url.match(/(?:youtu\.be\/|v=)([A-Za-z0-9_-]{11})/);
    return m ? m[1] : null;
  };

  const handleDownload = async () => {
    if (!videoUrl) return;
    setIsDownloading(true);
    setProgress(0);
    progressRef.current = setInterval(() =>
      setProgress(p => Math.min(p + Math.random() * 10, 99))
    , 300);

    try {
      const res = await axios.post('/youtube-download', { url: videoUrl });
      clearInterval(progressRef.current);
      setProgress(100);
      const vid = extractYouTubeId(videoUrl);
      const thumb = vid ? `https://img.youtube.com/vi/${vid}/hqdefault.jpg` : null;
      //const path = `http://localhost:5000${res.data.url}`;
      const mediaUrl = res.data.url; 
      console.log("res.data",res.data)
      
     const path = `http://${flaskHost}:5000${mediaUrl}`;


      
     setDownloadedVideo({ thumbnail: thumb, localUrl: path });
     setIsVideoRevealed(false);

     const a = document.createElement('a');
     a.href = path;
     a.download = 'downloaded.mp4';
     document.body.appendChild(a);
     a.click();
     document.body.removeChild(a);
    } catch {
      clearInterval(progressRef.current);
      alert('다운로드 중 오류가 발생했습니다');
    } finally {
      setTimeout(() => {
        setIsDownloading(false);
        setProgress(0);
      }, 500);
    }
  };

  const handleStartComparison = () => {
    alert('댄스 비교 기능을 시작합니다!');
    navigate('/upload');
  };

  return (
    <div >

      {/* 메인 컨텐츠 */}
      <main className="relative z-10 px-4 md:px-6 py-4 md:py-8">
        <div className="max-w-7xl mx-auto">
          {/* 제목 영역 */}
          <div className="text-center mb-12">
            <h1 className="text-3xl md:text-5xl lg:text-6xl font-bold text-white mb-4 animate-fade-in drop-shadow-lg">
             Move AI Scan
            </h1>
            <p className="text-sm md:text-lg text-white/90 max-w-5xl mx-auto drop-shadow-md">
              유튜브 영상을 다운로드하거나 직접 업로드해서 전문 안무와 비교 분석을 시작해보세요!
            </p>
            <div className="mt-6">
              <Sparkles className="w-8 h-8 text-yellow-200 animate-spin mx-auto" />
            </div>
          </div>

          {/* 카드 레이아웃 */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {/* 왼쪽: 다운로드 카드 */}
            <div className="p-6 bg-white/15 backdrop-blur-md border border-white/30 rounded-2xl shadow-2xl hover:shadow-3xl transition transform hover:scale-105">
              <h2 className="text-xl font-bold text-white mb-3 drop-shadow-md">
                영상 업로드 &amp; 다운로드
              </h2>
              <p className="text-sm text-white/80 mb-4">
                유튜브 URL을 입력하거나 직접 파일을 업로드하세요
              </p>
              <div className="flex space-x-3">
                <input
                  type="text"
                  placeholder="유튜브 영상 URL을 붙여넣기..."
                  value={videoUrl}
                  onChange={e => setVideoUrl(e.target.value)}
                  className="flex-grow h-10 px-3 rounded-lg bg-white/25 text-white placeholder-white/70 border border-white/40 focus:outline-none focus:ring-2 focus:ring-white/40"
                />
                {/* <button
                  onClick={handleDownload}
                  disabled={!videoUrl || isDownloading}
                  className="px-4 bg-gradient-to-r from-orange-400 to-pink-500 hover:from-orange-500 hover:to-pink-600 text-white rounded-lg shadow-lg disabled:opacity-50"
                >
                  {isDownloading
                    ? <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    : <DownloadIcon className="w-5 h-5" />
                  }
                </button> */}
                <button
                  onClick={handleDownload}
                  disabled={!videoUrl || isDownloading}
                  className="flex items-center px-4 bg-gradient-to-r from-orange-400 to-pink-500 hover:from-orange-500 hover:to-pink-600 text-white rounded-lg shadow-lg disabled:opacity-50"
                >
                  {isDownloading ? (
                    <>
                      <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                      <span className="ml-2 text-sm">{Math.floor(progress)}%</span>
                    </>
                  ) : (
                    <>
                      <DownloadIcon className="w-5 h-5" />
                      <span className="ml-2 text-sm">다운로드</span>
                    </>
                  )}
                </button>
              </div>

              {downloadedVideo && (
                <div className="mt-6">
                  {!isVideoRevealed ? (
                    <img
                      src={downloadedVideo.thumbnail}
                      alt="썸네일"
                      className="w-full h-48 object-cover rounded-lg cursor-pointer"
                      onClick={() => setIsVideoRevealed(true)}
                    />
                  ) : (
                    <video
                      src={downloadedVideo.localUrl}
                      controls
                      className="w-full rounded-lg shadow-md"
                    />
                  )}
                </div>
              )}
            </div>

            {/* 오른쪽: 비교 시작 카드 */}
            <div className="p-6 bg-white/15 backdrop-blur-md border border-white/30 rounded-2xl shadow-2xl hover:shadow-3xl transition transform hover:scale-105 flex flex-col justify-between">
              <div>
                <h2 className="text-xl font-bold text-white mb-3 drop-shadow-md">
                  댄서 vs 연습생 비교
                </h2>
                <p className="text-sm text-white/80 mb-4">
                  댄서와 연습생의 영상을 업로드하여 AI 기반 분석과 피드백으로 동작을 비교하세요
                </p>
              </div>
              <button
                onClick={handleStartComparison}
                className="w-full py-4 bg-gradient-to-r from-pink-400 to-rose-500 hover:from-pink-500 hover:to-rose-600 text-white font-bold rounded-xl shadow-lg transition"
              >
                댄스 비교 시작하기  <ArrowRight className="inline w-5 h-5 ml-2" />
              </button>
              <p className="mt-4 text-center text-white/80 text-xs">
                ✨ 전문가 수준의 분석으로 춤 실력을 향상시켜보세요!
              </p>
            </div>
          </div>
        </div>
      </main>



      {/* fade-in 애니메이션 */}
      <style>
        {`
          @keyframes fade-in {
            from { opacity: 0; transform: translateY(20px); }
            to   { opacity: 1; transform: translateY(0); }
          }
          .animate-fade-in {
            animation: fade-in 0.6s ease-out forwards;
          }
        `}
      </style>
    </div>
  );
}

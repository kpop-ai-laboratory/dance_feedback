import React, { useState, useRef, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { ArrowLeft, Download, FileText } from 'lucide-react';

export default function SequenceReviewPage() {
  const navigate = useNavigate();
  const { state } = useLocation();

  // 비디오 URL
  const videoUrl = state.finalVideo.startsWith('/data')
    ? state.finalVideo
    : `/data/${state.finalVideo}`;
  // 전체 시퀀스 피드백 리스트
  const seqFeedbacks = state.sequence_feedbacks || [];

  // 누적 표시되는 피드백 리스트
  const [shownList, setShownList] = useState([]);
  const shownRef = useRef(new Set());
  const listContainerRef = useRef(null);

  const videoRef = useRef(null);

  const handleTimeUpdate = () => {
    if (!videoRef.current) return;
    const t = videoRef.current.currentTime;
    // const matched = seqFeedbacks.find(
    //   ({ start_time, end_time }) =>
    //     t >= start_time && t <= end_time && !shownRef.current.has(start_time)
    // );
    const matched = seqFeedbacks.find(
      ({ end_time }) =>
        t >= end_time && !shownRef.current.has(end_time)
    );
    if (matched) {
      shownRef.current.add(matched.end_time);
      setShownList(prev => [matched, ...prev]);
    }
  };

  // 피드백이 추가될 때마다 스크롤 맨 위로
  useEffect(() => {
    console.log("seqFeedbacks", seqFeedbacks)
    if (listContainerRef.current) {
      listContainerRef.current.scrollTop = 0;
    }
  }, [shownList]);

  // 다운로드 상태
  const [isDownloadingVideo, setIsDownloadingVideo] = useState(false);
  const [isDownloadingJSON, setIsDownloadingJSON] = useState(false);

  const downloadVideo = () => {
    setIsDownloadingVideo(true);
    setTimeout(() => {
      setIsDownloadingVideo(false);
      const a = document.createElement('a');
      a.href = videoUrl;
      a.download = 'dance_feedback.mp4';
      a.click();
    }, 2000);
  };

  const downloadJSON = () => {
    setIsDownloadingJSON(true);
    setTimeout(() => {
      setIsDownloadingJSON(false);
      const report = {
        generated_at: new Date().toISOString(),
        feedbacks: shownList,
      };
      const blob = new Blob([JSON.stringify(report, null, 2)], {
        type: 'application/json',
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'feedback_report.json';
      a.click();
      URL.revokeObjectURL(url);
    }, 2000);
  };

  const goBack = () => navigate(-1);
  const isEmpty = shownList.length === 0;

  return (
     <div className="text-white">
      

      <main className="flex-grow flex flex-col items-center px-4 py-8">
        <button onClick={goBack} className="self-start mb-4 flex items-center hover:underline">
          <ArrowLeft className="mr-2" /> 뒤로
        </button>

        <h1 className="text-3xl font-bold mb-6">🎬 Sequence Feedback</h1>

        {/* 순수 비디오 */}
        <div className="w-full max-w-4xl mb-8">
          <video
            ref={videoRef}
            src={videoUrl}
            controls
            onTimeUpdate={handleTimeUpdate}
            className="w-full rounded-lg shadow-lg"
          />
        </div>

        
        {/* 아래 피드백 목록: 재생 타이밍에 따라 하나씩 추가 */}
      <div className="w-full max-w-4xl mb-8">
          <h2 className="text-xl font-semibold mb-2">📋 전체 피드백 목록</h2>
          <p className="text-xs text-white/60 mb-2">
            ※ 시퀀스 피드백은 동작 전환의 속도와 큼지막한 움직임을 비교하여 생성됩니다.
          </p>
          <div
            ref={listContainerRef}
            className="bg-white/15 backdrop-blur-md border border-white/30 rounded-2xl max-h-96 overflow-y-auto p-4 space-y-4 shadow-inner"
          >
            {isEmpty ? (
              <p className="text-center text-white/70">피드백이 없습니다.</p>
            ) : (
              shownList.map((f, i) => (
                <div key={i} className="p-3 bg-white/25 rounded-lg text-white text-sm">
                  <span className="text-xs font-semibold">
                    {new Date(f.end_time * 1000).toISOString().substr(11, 8)}
                  </span>
                  <div className="mt-1">
                    {f.feedback.split('\n').map((line, i) => (
                      <p key={i}>{line}</p>
                    ))}
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* 다운로드 버튼 */}
        <div className="flex flex-col md:flex-row gap-4 w-full max-w-4xl">
          <button
            onClick={downloadVideo}
            disabled={isDownloadingVideo}
            className="flex-1 flex items-center justify-center py-3 bg-gradient-to-r from-indigo-400 to-purple-500 hover:from-indigo-500 hover:to-purple-600 rounded-2xl font-bold disabled:opacity-50"
          >
            {isDownloadingVideo ? '다운로드 중…' : <><Download className="mr-2" />동영상 다운로드</>}
          </button>
          <button
            onClick={downloadJSON}
            disabled={isDownloadingJSON}
            className="flex-1 flex items-center justify-center py-3 bg-gradient-to-r from-blue-400 to-cyan-500 hover:from-blue-500 hover:to-cyan-600 rounded-2xl font-bold disabled:opacity-50"
          >
            {isDownloadingJSON ? '다운로드 중…' : <><FileText className="mr-2" />JSON 다운로드</>}
          </button>
        </div>
      </main>

     </div>
  );
}

import React, { useState, useEffect, useRef } from 'react';
// 등급별 아이콘
import perfectImg from '../assets/Perfect.png';
import greatImg   from '../assets/Great.png';
import goodImg    from '../assets/Good.png';
import badImg     from '../assets/Bad.png';

const formatTime = sec => {
  const m = String(Math.floor(sec/60)).padStart(2,'0');
  const s = String(sec % 60).padStart(2,'0');
  return `${m}:${s}`;
};
const getGrade = score => {
  const pct = score * 100;
  if (pct >= 90) return { label: 'Perfect', icon: perfectImg };
  if (pct >= 80) return { label: 'Great',   icon: greatImg   };
  if (pct >= 70) return { label: 'Good',    icon: goodImg    };
  return { label: 'Bad',     icon: badImg     };
};

export default function TestViewer() {
  const jobId = "9534d99982174d68b0f842c5a06a7954";

  // --- 상태 정의
  const [frameScores,  setFrameScores]  = useState([]);  // 프레임별 점수
  const [secScores,    setSecScores]    = useState([]);  // 초별 평균 점수
  const [lowestInSec,  setLowestInSec]  = useState([]);  // 초별 최저 프레임 인덱스
  const [fps,          setFps]          = useState(30);
  const [feedback,     setFeedback]     = useState({});
  const [history,      setHistory]      = useState([]);
  const [currentGrade, setCurrentGrade] = useState({ label:'', icon:'' });

  const videoRef = useRef();
  const threshold = 0.8;

  useEffect(() => {
    // scores.json 에서 frame_scores/second_scores/second_lowest_frames/fps 가져오기
    fetch(`/data/${jobId}/dancer_kp/scores.json`)
      .then(r => r.json())
      .then(j => {
        setFrameScores(j.frame_scores   || []);
        setSecScores(  j.second_scores  || []);
        setLowestInSec(j.second_lowest_frames || []);
        setFps( j.fps || 30);
      });
    // feedback.json
    fetch(`/data/${jobId}/dancer_kp/feedback.json`)
      .then(r => r.json())
      .then(j => setFeedback(j));
  }, []);

  const onTimeUpdate = () => {
    if (!videoRef.current) return;
    const sec      = videoRef.current.currentTime;
    const frameIdx = Math.floor(sec * fps);
    const secIdx   = Math.floor(sec);

    // 1) 실시간 Grade 갱신 (프레임별)
    const simF = frameScores[frameIdx];
    if (simF !== undefined) {
      setCurrentGrade(getGrade(simF));
    }

    // 2) “초별 평균”이 임계치 이하면 → 히스토리 추가
    const avg = secScores[secIdx];
    if (avg === undefined || avg >= threshold) return;

    // 3) 이 초에서 가장 나빴던 프레임
    const badFrame = lowestInSec[secIdx];
    if (badFrame === undefined) return;

    // 4) 해당 프레임의 피드백
    const msgs = feedback[badFrame] || [];
    if (!msgs.length) return;
    // (0° 제거 필터)
    const valid = msgs.filter(m => {
      const tea = m.match(/선생님\s*([\d.]+)°/);
      const stu = m.match(/학생\s*([\d.]+)°/);
      return tea && stu && parseFloat(tea[1])>0 && parseFloat(stu[1])>0;
    });
    if (!valid.length) return;

    // 5) 중복 방지 & 히스토리 추가
    const ts  = formatTime(secIdx);
    const key = `${ts}#${badFrame}`;
    if (history.some(e => e.key === key)) return;

    setHistory(h => [
      ...h,
      { key, time: ts, frame: badFrame, msgs: valid }
    ]);
  };

  // 히스토리 클릭 시 해당 프레임으로 이동
  const seekTo = entry => {
    if (!videoRef.current) return;
    videoRef.current.currentTime = entry.frame / fps;
    videoRef.current.play();
  };

  return (
    <div className="p-8 flex flex-col items-center space-y-6">
      {/* ▶ 동영상 + 실시간 Grade */}
      <div className="relative w-full max-w-2xl">
        <video
          ref={videoRef}
          src={`/data/${jobId}/final_feedback_with_audio.mp4`}
          controls
          onTimeUpdate={onTimeUpdate}
          className="w-full rounded border"
        />
        {currentGrade.label && (
          <div className="absolute top-2 left-2 flex items-center bg-white bg-opacity-75 p-2 rounded shadow-lg z-10">
            <img src={currentGrade.icon}
                 alt={currentGrade.label}
                 className="w-6 h-6 mr-2"/>
            <span className="font-medium">{currentGrade.label}</span>
          </div>
        )}
      </div>

      {/* 📝 피드백 히스토리 */}
      {history.length > 0 && (
        <div className="w-full max-w-2xl bg-yellow-100 rounded shadow p-4 max-h-80 overflow-y-auto">
          <h3 className="font-semibold mb-2">📋 피드백 히스토리</h3>
          {history.map(entry => (
            <div
              key={entry.key}
              onClick={() => seekTo(entry)}
              className="mb-4 p-2 cursor-pointer hover:bg-yellow-200 rounded transition-colors"
            >
              <div className="flex items-center">
                <span className="underline text-sm font-medium mr-2">{entry.time}</span>
                <span className="text-xs text-gray-600 ml-auto">frame #{entry.frame}</span>
              </div>
              <ul className="list-disc list-inside text-sm ml-4 mt-1">
                {entry.msgs.map((m,i) => <li key={i}>{m}</li>)}
              </ul>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// // src/components/CompareResult.js
// import React, { useState, useEffect, useRef } from 'react';
// import { useLocation, useNavigate } from 'react-router-dom';
// import perfectImg   from '../assets/Perfect.png';
// import greatImg     from '../assets/Great.png';
// import goodImg      from '../assets/Good.png';
// import badImg       from '../assets/Bad.png';
// import DownloadIcon from '../assets/Download.png';

// // 시간 포맷 (MM:SS)
// const formatTime = sec => {
//   const m = String(Math.floor(sec/60)).padStart(2,'0');
//   const s = String(Math.floor(sec % 60)).padStart(2,'0');
//   return `${m}:${s}`;
// };

// // 0~1 점수를 4단계 Grade로 변환
// const getGrade = score => {
//   const pct = score * 100;
//   if (pct >= 90) return { label: 'Perfect', icon: perfectImg };
//   if (pct >= 80) return { label: 'Great',   icon: greatImg   };
//   if (pct >= 70) return { label: 'Good',    icon: goodImg    };
//   return { label: 'Bad',     icon: badImg     };
// };

// export default function CompareResult() {
//   const location = useLocation();
//   const navigate = useNavigate();
//   const state    = location.state;  // 여기서만 조건체크

//   // ── Hooks는 절대 조건문 밖 최상단에──
//   const videoRef       = useRef();
//   const [fps, setFps]                   = useState(30);
//   const [frameScores, setFrameScores]   = useState({});
//   const [secScores, setSecScores]       = useState({});
//   const [lowestInSec, setLowestInSec]   = useState({});
//   const [feedback, setFeedback]         = useState({});
//   const [history, setHistory]           = useState([]);
//   const [currentGrade, setCurrentGrade] = useState(null);

//   const threshold = 0.8;

//   // 잘못된 접근 시 리다이렉트
//   useEffect(() => {
//     if (!state) {
//       navigate('/', { replace: true });
//     }
//   }, [state, navigate]);

//   // scores.json / feedback.json 불러오기
//   useEffect(() => {
//     if (!state) return;  
//     const { scoresJson, feedbackJson } = state;

//     fetch(scoresJson)
//       .then(r => r.json())
//       .then(j => {
//         console.log("1",j);
//         setFps(j.fps);
//         setFrameScores(j.frame_scores);
//         setSecScores(j.second_scores);
//         setLowestInSec(j.second_lowest_frames);
//       });

//     fetch(feedbackJson)
//       .then(r => r.json())
//       .then(j =>{ 
//         setFeedback(j)
//         console.log("feedback",feedback);
//   });
//   }, [state]);

//   // 재생 중 Grade 갱신 & 히스토리 구축
//   const onTimeUpdate = () => {
//     if (!videoRef.current || !state) return;
//     const sec      = videoRef.current.currentTime;
//     const frameIdx = Math.floor(sec * fps);
//     const secIdx   = Math.floor(sec);

//     // 1) 초별 평균 → Grade
//     const avg = secScores[secIdx];
//     if (avg != null) {
//       setCurrentGrade(getGrade(avg));
//     }

//     // 2) 평균 < threshold 시 히스토리 추가
//     if (avg === undefined || avg >= threshold) return;
//     const badFrame = lowestInSec[secIdx];
//     if (badFrame === undefined) return;
//     const msgs = feedback[badFrame] || [];
//     if (!msgs.length) return;

//     const key = `${secIdx}#${badFrame}`;
//     if (history.some(e => e.key === key)) return;

//     setHistory(h => [
//       ...h,
//       { key, time: formatTime(secIdx), frame: badFrame, msgs }
//     ]);
//   };

//   // 전체 JSON 다운로드
//   const handleDownload = () => {
//     if (!state) return;
//     const data = JSON.stringify({ frameScores, feedback }, null, 2);
//     const blob = new Blob([data], { type: 'application/json' });
//     const url  = URL.createObjectURL(blob);
//     const a    = document.createElement('a');
//     a.href     = url;
//     a.download = 'scores_feedback.json';
//     a.click();
//     URL.revokeObjectURL(url);
//   };

//   // 히스토리 클릭 시 해당 프레임으로 이동
//   const seekTo = entry => {
//     if (!videoRef.current) return;
//     videoRef.current.currentTime = entry.frame / fps;
//     videoRef.current.play();
//   };

//   // state 없으면 렌더링 안 함 (redirect effect가 실행됨)
//   if (!state) return null;

//   const { finalVideo, durations } = state;

//   return (
//     <div className="flex space-x-8 px-8 py-4">
//       {/* 왼쪽: 결과 비디오 (2/3) */}
//       <div className="w-2/3">
//         <video
//           ref={videoRef}
//           src={finalVideo}
//           controls
//           onTimeUpdate={onTimeUpdate}
//           className="w-full h-auto rounded-lg border"
//         />
//       </div>

//       {/* 오른쪽: Grade / 히스토리 / 다운로드 (1/3) */}
//       <div className="w-1/3 flex flex-col space-y-4">
//         {/* Grade 표시 */}
//         <div className="text-right">
//           {currentGrade && (
//             <div className="inline-flex items-center bg-white bg-opacity-90 p-2 rounded-full shadow">
//               <img
//                 src={currentGrade.icon}
//                 alt={currentGrade.label}
//                 className="w-6 h-6 mr-2"
//               />
//               {/* <span className="font-medium text-lg">
//                 {currentGrade.label}
//               </span> */}
//             </div>
//           )}
//         </div>

//         {/* 피드백 히스토리 (스크롤 가능) */}
//         <div className="flex-1 max-h-80 overflow-y-auto bg-yellow-50 p-4 rounded shadow-sm">
//           <h3 className="font-semibold mb-2">📋 피드백 히스토리</h3>
//           {history.length === 0 ? (
//             <p className="text-sm text-gray-500">이상 구간이 없습니다.</p>
//           ) : (
//             history.map(entry => (
//               <div
//                 key={entry.key}
//                 onClick={() => seekTo(entry)}
//                 className="mb-4 p-2 cursor-pointer hover:bg-yellow-200 rounded transition-colors"
//               >
//                 <div className="flex items-center">
//                   <span className="underline text-sm font-medium mr-2">
//                     {entry.time}
//                   </span>
//                   <span className="text-xs text-gray-600 ml-auto">
//                     frame #{entry.frame}
//                   </span>
//                 </div>
//                 <ul className="list-disc list-inside text-sm ml-4 mt-1">
//                   {entry.msgs.map((m,i) => <li key={i}>{m}</li>)}
//                 </ul>
//               </div>
//             ))
//           )}
//         </div>

//         {/* 전체 다운로드 버튼 */}
//         <div className="text-right">
//           <button
//             onClick={handleDownload}
//             className="inline-flex items-center px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition"
//           >
//             <img src={DownloadIcon} alt="" className="w-4 h-4 mr-2" />
//             전체 다운로드
//           </button>
//         </div>
//       </div>
//     </div>
//   );
// }


// src/components/CompareResult.jsx
import React, { useState, useEffect, useRef } from 'react';
import { useLocation, useNavigate }          from 'react-router-dom';
import GradeBadge from './GradeBadge';
import perfectImg   from '../assets/Perfect.png';
import greatImg     from '../assets/Great.png';
import goodImg      from '../assets/Good.png';
import badImg       from '../assets/Bad.png';
import DownloadIcon from '../assets/Download.png';

// MM:SS 포맷
const formatTime = sec => {
  const m = String(Math.floor(sec/60)).padStart(2,'0');
  const s = String(Math.floor(sec % 60)).padStart(2,'0');
  return `${m}:${s}`;
};
// 0~1 → 4단계
const getGrade = score => {
  const pct = score * 100;
  if (pct >= 90) return { label: 'Perfect', icon: perfectImg };
  if (pct >= 80) return { label: 'Great',   icon: greatImg   };
  if (pct >= 70) return { label: 'Good',    icon: goodImg    };
  return { label: 'Bad',     icon: badImg     };
};

export default function CompareResult() {
  ////////////////////////////////////////
  // 1) Hook은 항상 최상단에만 선언
  const location = useLocation();
  const navigate = useNavigate();
  const videoRef = useRef();

  const [fps, setFps]                   = useState(30);
  const [frameScores, setFrameScores]   = useState({});
  const [secScores, setSecScores]       = useState({});
  const [lowestInSec, setLowestInSec]   = useState({});
  const [feedback, setFeedback]         = useState({});
  const [chunkAvgs, setChunkAvgs]       = useState([]);
  const [currentGrade, setCurrentGrade] = useState(null);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [currentFrameFeedback, setCurrentFrameFeedback] = useState([]);
  const [history, setHistory]           = useState([]);
  const [tab, setTab]                   = useState('feedback');
  const [showPanel, setShowPanel]       = useState(true);

  const threshold = 0.8;

  ////////////////////////////////////////
  // 2) guard & JSON 로딩
  const state = location.state;
  useEffect(() => {
    if (!state) {
      navigate('/', { replace: true });
      return;
    }
    const { scoresJson, feedbackJson } = state;

    // scores.json 읽어서 각종 점수 세팅
    fetch(scoresJson)
      .then(r => r.json())
      .then(j => {
        console.log(j)
        setFps(j.fps);
        setFrameScores(j.frame_scores);
        setSecScores(j.second_scores);
        setLowestInSec(j.second_lowest_frames);
        
        // chunkAvgs: fps/3 프레임 묶음 평균
        const cs   = Math.floor(j.fps / 3) || 1;
        const vals = Object.values(j.frame_scores).map(Number);
        const avgs = [];
        for (let i = 0; i < vals.length; i += cs) {
          const slice = vals.slice(i, i + cs);
          avgs.push(slice.reduce((a,b)=>a+b,0)/slice.length);
        }
        setChunkAvgs(avgs);
      });

    fetch(feedbackJson)
      .then(r => r.json())
      .then(j => setFeedback(j));
  }, [state, navigate]);

  if (!state) return null;  
  const { finalVideo } = state;

  ////////////////////////////////////////
  // 3) 재생 중 업데이트
  const onTimeUpdate = () => {
    if (!videoRef.current) return;
    if (!showPanel) setShowPanel(true);

     // 1) 현재 재생 위치(sec → frameIdx)  
    const sec      = videoRef.current.currentTime;
    const frameIdx = Math.round(sec * fps);
    setCurrentFrame(frameIdx);
    setCurrentFrameFeedback(feedback[frameIdx] || []);

    // 1초에 fps/3번 Grade
    const cs  = Math.floor(fps/3) || 1;
    const idx = Math.floor(frameIdx / cs);
    const avg = chunkAvgs[idx];
    if (avg != null) setCurrentGrade(getGrade(avg));

      // if second-average < threshold, add the worst frame of that second
      const secIdx = Math.floor(sec);
      const secAvg = secScores[secIdx];
      if (secAvg !== undefined && secAvg < threshold) {
        const badFrame = lowestInSec[secIdx];
        const msgs     = feedback[badFrame] || [];
        if (msgs.length) {
          const key = `${secIdx}#${badFrame}`;
          if (msgs.length && !history.some(e=>e.key===`${secIdx}#${badFrame}`)) {
            setHistory(h => [
              ...h,
              {
                key:`${secIdx}#${badFrame}`,
                time: formatTime(secIdx),
                frame: badFrame,
                msgs
              }
            ]);
          }
        }
      }    
  };



  ////////////////////////////////////////
// 4) 클릭 이동
const seekTo = entry => {
  if (!videoRef.current) return;

  // 1) 우선 비디오를 멈추고
  videoRef.current.pause();

  //정확히 그 프레임의 시간(sec)으로 이동
  const targetTime = entry.frame / fps;
  videoRef.current.currentTime = targetTime;

  // 3) seek가 완전히 일어난 뒤에 한 번 더 pause
  //    (seeked 이벤트에서 강제 멈춤)
  // 3) 클릭한 대로 UI도 즉시 갱신
  setCurrentFrame(entry.frame);
  setCurrentFrameFeedback(entry.msgs);
};
  ////////////////////////////////////////
  // 5) 다운로드
  const handleDownload = () => {
    const data = JSON.stringify({ frameScores, feedback }, null, 2);
    const blob = new Blob([data], { type: 'application/json' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = 'scores_feedback.json';
    a.click();
    URL.revokeObjectURL(url);
  };

  ////////////////////////////////////////
  // 6) 렌더링
  return (
    <div className="flex h-[80vh]">
      {/* 왼쪽: 영상 (1/2) */}
      <div className="w-1/2 p-4">
        <video
          ref={videoRef}
          src={finalVideo}
          controls
          onTimeUpdate={onTimeUpdate}
          className="w-full h-full object-contain rounded-lg border"
        />
      </div>

      {showPanel && (
        // <div className="w-1/2 flex flex-col space-y-4 p-4">
        <div className="w-1/2 flex flex-col h-full p-4">
          {/* 닫기 */}
          <div className="flex justify-end">
            <button 
              onClick={()=>setShowPanel(false)} 
              className="text-2xl font-bold"
            >×</button>
          </div>

          {/* Grade
          <div className="flex justify-center">
            {currentGrade && (
              <div className="inline-flex items-center bg-white p-3 rounded-full shadow-lg animate-pulse">
                <img
                  src={currentGrade.icon}
                  alt={currentGrade.label}
                  className="w-12 h-12 mr-2"
                />
              </div>
            )}
          </div> */}

          <div className="flex justify-center">
            { currentGrade && (
              <GradeBadge
                label={currentGrade.label}
                color={currentGrade.label.toLowerCase()} 
              />
            )}
          </div>

          {/* Frame Feedback */}
          <div className="bg-white p-4 rounded shadow-sm h-1/3">
          {/* <div className="sticky top-0 z-20 bg-white p-4 rounded shadow mb-4"> */}
            <h4 className="font-semibold mb-2">Frame {currentFrame} Feedback</h4>
            {currentFrameFeedback.length === 0
              ? <p className="text-gray-500">피드백이 없습니다.</p>
              : (
                <ul className="list-disc list-inside text-sm ml-4">
                  {currentFrameFeedback.map((m,i)=>(
                    <li key={i}>{m}</li>
                  ))}
                </ul>
              )}
          </div>

          {/* 탭 */}
          <div className="flex justify-center space-x-4">
            {['feedback','all'].map(x=>(
              <button
                key={x}
                onClick={()=>setTab(x)}
                className={`px-4 py-1 rounded ${
                  tab===x
                    ? 'bg-red-300 text-white'
                    : 'bg-gray-200 text-gray-700'
                }`}
              >
                {x==='feedback' ? 'FeedBack History' : 'All History'}
              </button>
            ))}
          </div>

          {/* 히스토리 리스트 (스크롤) */}
            {/* 히스토리 스크롤 영역 */}
<div className="flex-1 overflow-y-auto bg-yellow-50 p-4 rounded shadow-sm">
  <h3 className="font-semibold mb-2">📋 피드백 히스토리</h3>

  {tab === 'feedback' ? (
    history.length === 0
      ? <p className="text-gray-500">이상 구간이 없습니다.</p>
      : history.map(entry => (
        <div
        key={entry.key}
        onClick={() => seekTo(entry)}
        className="mb-4 p-2 cursor-pointer hover:bg-yellow-200 rounded transition-colors"
      >
        <div className="flex items-center">
          <span className="underline text-sm font-medium mr-2">
            {entry.time}
          </span>
          <span className="text-xs text-gray-600 ml-auto">
            frame #{entry.frame}{'  '}{'  '}{'  '}
            {/* score {(score * 100).toFixed(1)}% */}
            score {(frameScores[entry.frame] * 100).toFixed(1)}%
          </span>
        </div>
        <ul className="list-disc list-inside text-sm ml-4 mt-1">
          {entry.msgs.map((m,i) => (
            <li key={i}>{m}</li>
          ))}
        </ul>
      </div>
        ))
  ) : (
    Object.entries(frameScores).map(([frame, score]) => {
      const msgs = feedback[frame] || ['피드백 없음'];
      return (
        <div
          key={frame}
          onClick={() => seekTo({ frame: Number(frame), msgs })}
          className="mb-4 p-2 cursor-pointer hover:bg-yellow-200 rounded"
        >
          <div className="flex items-center">
            <span className="text-sm font-medium">
              frame #{frame}
            </span>
            <span className="text-xs text-gray-600 ml-auto">
              {/* score {(score * 100).toFixed(1)}% */}
              {/* frame #{entry.frame} */}
            </span>
          </div>
          <ul className="list-disc list-inside text-sm ml-4 mt-1">
            {msgs.map((m, i) => <li key={i}>{m}</li>)}
          </ul>
        </div>
      );
    })
  )}
</div>

          {/* 다운로드 */}
          <div className="flex justify-center">
            <button
              onClick={handleDownload}
              className="inline-flex items-center px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
            >
              <img src={DownloadIcon} alt="" className="w-4 h-4 mr-2"/>
              전체 다운로드
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

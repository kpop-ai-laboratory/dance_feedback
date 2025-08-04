import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import VideoUploadCard from '../components/VideoUploadCard';
import LoadingOverlay  from '../components/LoadingOverlay';

export default function Upload2() {
  const [dancer, setDancer]           = useState(null);
  const [trainee, setTrainee]         = useState(null);
  const [isLoading, setIsLoading]     = useState(false);

  const navigate = useNavigate();

  const handleStartCompare = async () => {
    if (!dancer || !trainee) return;
    setIsLoading(true);

    const form = new FormData();
    form.append('dancer', dancer);
    form.append('trainee', trainee);

    try {
      const res = await axios.post(
        '/compare/',
        form,
        { headers: { 'Content-Type': 'multipart/form-data' } }
      );

      // 서버에서 받은 결과 경로
      const finalVideo   = `/data/${res.data.final_video}`;
      const feedbackJson = `/data/${res.data.feedback_json}`;
      const scoresJson   = `/data/${res.data.scores_json}`;
      const durations    = res.data.durations;

      // 결과 페이지로 이동하며 state 전달
      navigate('/result', {
        state: { finalVideo, feedbackJson, scoresJson, durations }
      });
    } catch (err) {
      console.error(err);
      alert('서버 호출 중 에러가 발생했습니다');
    } finally {
      setIsLoading(false);
    }
  };


  const handleFeedback = async () => {
    if (!dancer || !trainee) return;      // 파일 둘 다 없으면 무시
    setIsLoading(true);            // 로딩 오버레이 표시

    const form = new FormData();
    form.append('dancer', dancer);
    form.append('trainee', trainee);

    try {
      // 백엔드 /compared_by_seq/ 엔드포인트 호출
      const res = await axios.post('/compared_by_seq/', form, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      // // 👉 서버가 보내주는 데이터 확인
      // console.log('handleFeedback res.data:', res.data);

      // // 실제 백엔드가 { final_video: ".../merged.mp4" } 형태로 반환하므로
      // const videoUrl = `/data/${res.data.final_video}`;

      // 서버가 준 final_video: "<jobId>/merged.mp4"
      const { final_video, sequence_feedbacks } = res.data;  // API 응답에서 실제 feedbacks 꺼내오기
      const videoUrl = `/data/${final_video}`;
      const jobId = final_video.split('/')[0];

      navigate('/sequence-result', {
        state: {
          jobId,
          finalVideo: videoUrl,
          sequence_feedbacks   // ✅ 여기 실제 리스트를 넘겨줍니다
        }
      });
    } catch (err) {
      console.error(err);
      alert('평가 영상 피드백 중 오류가 발생했습니다.');
    } finally {
      setIsLoading(false);
    }
  };


  return (
    <main className="flex-grow flex items-center justify-center px-20">
        <div className="w-full max-w-5xl mt-24 p-14 bg-white/15 
        backdrop-blur-md border border-white/30 rounded-2xl shadow-2xl">
          <h1 className="text-3xl md:text-4xl font-bold text-white mb-8 drop-shadow-md">
            영상 업로드
          </h1>
          
          <div className="grid md:grid-cols-2 gap-8 mb-8">
          <VideoUploadCard
            label="댄서 영상 업로드"
            file={dancer}
            onChange={file => setDancer(file)}
            inputId="dancer-input"
            className="bg-transparent text-white drop-shadow-md"
          />
        
          <VideoUploadCard
            label="연습생 영상 업로드"
            file={trainee}
            onChange={file => setTrainee(file)}
            inputId="trainee-input"
            className="bg-transparent text-white drop-shadow-md"
          />
        </div>
      

          {/* 두 개 버튼 */}
          <div className="flex flex-col md:flex-row gap-4 mb-6">
            <button
              onClick={handleStartCompare}
              disabled={!dancer || !trainee || isLoading}
              className="flex-1 py-4 bg-gradient-to-r from-pink-400 to-rose-500 hover:from-pink-500 hover:to-rose-600 text-white font-bold rounded-2xl shadow-lg disabled:opacity-50 transition drop-shadow-md"
            >
              {isLoading ? '비교 중…' : '동시 비교하기'}
            </button>
            <button
              onClick={handleFeedback}
              disabled={!dancer || isLoading}
              className="flex-1 py-4 bg-gradient-to-r from-indigo-400 to-purple-500 hover:from-indigo-500 hover:to-purple-600 text-white font-bold rounded-2xl shadow-lg disabled:opacity-50 transition drop-shadow-md"
            >
              평가 영상 피드백
            </button>
          </div>
          
        </div>

      <LoadingOverlay
        isLoading={isLoading}
        text="댄스 비교 피드백 생성 중 입니다."
      />
  </main>
  );
}

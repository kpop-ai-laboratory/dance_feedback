import React from 'react';
import { Camera, List, Music, MessageCircle, Clock } from 'lucide-react';

export default function About() {
  return (
    <div className="max-w-5xl mx-auto px-6 py-12 text-white space-y-12">
      {/* 1. 페이지 헤더 및 소개 */}
      <header className="space-y-4 text-center">
        <h1 className="text-4xl font-extrabold">About Move AI Scan</h1>
        <p className="text-lg">
          안무 선생님의 동작을 실시간으로 정밀 모션 캡처하고, <br />
          음악 흐름에 따른
          자동 점수 평가 및 피드백 캡션 생성으로 연습생과 아티스트에게
          정확한 피드백을 제공합니다.
        </p>
      </header>

      {/* 2. 우리 프로젝트의 목표 */}
      <section className="bg-white/10 backdrop-blur-md rounded-2xl p-6">
        <h2 className="text-2xl font-semibold mb-2">Our Mission</h2>
        <p className="text-sm">
          Move AI Scan은 딥러닝 기반 모션 캡처 기술을 활용해 안무를 분석하고,
          학습자에게 실시간 피드백을 제공하여 연습 효율을 극대화하는 것을 목표로 합니다.
        </p>
      </section>

      {/* 3. 핵심 기능 */}
      <section>
        <h2 className="text-2xl font-semibold mb-6 text-center">핵심 기능</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {/* 실시간 모션 캡처 */}
          <div className="bg-white/20 backdrop-blur-md rounded-2xl p-6 shadow-lg flex flex-col h-full">
            <div className="flex items-center mb-3">
              <Camera className="w-6 h-6 text-yellow-300 mr-2" />
              <h3 className="text-xl font-semibold">실시간 모션 캡처</h3>
            </div>
            <p className="flex-1 text-sm mb-4">
              MediaPipe 기반 스켈레톤 키포인트를 실시간으로 추출하여
              지연 없이 모션을 기록합니다.
            </p>
            <ul className="list-disc list-inside text-xs space-y-1">
              <li>고정밀 키포인트 검출</li>
              <li>최소 지연 데이터 파이프라인</li>
            </ul>
          </div>

          {/* 자동 점수 평가 */}
          <div className="bg-white/20 backdrop-blur-md rounded-2xl p-6 shadow-lg flex flex-col h-full">
            <div className="flex items-center mb-3">
              <Music className="w-6 h-6 text-green-300 mr-2" />
              <h3 className="text-xl font-semibold">자동 점수 평가</h3>
            </div>
            <p className="flex-1 text-sm mb-4">
              안무 타이밍과 정확도를 음악 비트에 맞춰 AI가 점수화합니다.
            </p>
            <ul className="list-disc list-inside text-xs space-y-1">
              <li>프레임별 오차율 분석</li>
              <li>비트 매칭 최적화</li>
            </ul>
          </div>

          {/* 피드백 캡션 생성 */}
          <div className="bg-white/20 backdrop-blur-md rounded-2xl p-6 shadow-lg flex flex-col h-full">
            <div className="flex items-center mb-3">
              <MessageCircle className="w-6 h-6 text-purple-300 mr-2" />
              <h3 className="text-xl font-semibold">피드백 캡션 생성</h3>
            </div>
            <p className="flex-1 text-sm mb-4">
              점수 결과를 바탕으로 개선 포인트를 자동 캡션 형태로 제공합니다.
            </p>
            <ul className="list-disc list-inside text-xs space-y-1">
              <li>맞춤형 코멘트 자동 출력</li>
              <li>역사적 히스토리 비교 기능</li>
            </ul>
          </div>
        </div>
      </section>

      {/* 4. 사용 방법 */}
      <section>
        <h2 className="text-2xl font-semibold mb-6 text-center">How to Use?</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-10">
          {/* 1) 동시 비교하기 */}
          <div className="bg-white/20 backdrop-blur-md rounded-2xl p-6 shadow-lg flex flex-col h-full">
            <div className="flex items-center mb-2">
            <List className="w-6 h-6 text-purple-300 mr-2" />
              <h3 className="text-xl font-semibold">동시 비교하기</h3>
            </div>
            <p className="text-sm text-white/70 mb-4">프레임 기반 피드백</p>
            <p className="flex-1 text-sm mb-4">
              두 영상을 같은 타임라인에서 동기화해
              프레임별 오차를 수치로 분석합니다.
            </p>
            <ul className="list-disc list-inside text-xs space-y-1">
              <li>정밀 자세 비교</li>
              <li>원하는 순간 캡처</li>
            </ul>
          </div>

          {/* 2) 평가 영상 피드백 */}
          <div className="bg-white/20 backdrop-blur-md rounded-2xl p-6 shadow-lg flex flex-col h-full">
            <div className="flex items-center mb-2">
              <List className="w-6 h-6 text-purple-300 mr-2" />
              <h3 className="text-xl font-semibold">평가 영상 피드백</h3>
            </div>
            <p className="text-sm text-white/70 mb-4">시퀀스 기반 피드백</p>
            <p className="flex-1 text-sm mb-4">
              전체 안무를 파트별로 분할해
              대표 프레임으로 자동 피드백을 생성합니다.
            </p>
            <ul className="list-disc list-inside text-xs space-y-1">
              <li>파트별 평균 오차 한눈에 파악</li>
              <li>Feedback History 조회</li>
            </ul>
          </div>
        </div>
      </section>

      {/* 5. 팁 & 트릭 */}
      <section>
        <div className="bg-white/20 backdrop-blur-md rounded-2xl p-6 shadow-lg">
          <div className="flex items-center mb-4">
            <Clock className="w-6 h-6 text-pink-300 mr-2" />
            <h3 className="text-2xl font-semibold">팁 & 트릭</h3>
          </div>
          <ul className="list-disc list-inside space-y-2 text-sm">
            <li>영상 길이는 5분 이내로 업로드하는 걸 추천합니다.</li>
            <li>촬영 시 화면 앞에서 2~3미터 거리, 전신이 프레임에 들어오도록.</li>
          </ul>
        </div>
      </section>
    </div>
  );
}

1. 유현모

Multi-vairableLinearRegression 에 대한 실습 진행과 더불어서 필요한 파이썬 기능,        이론 학습 

CSV(Comma-separated values)의 약자로서 CSV 파일은 각 라인의 컬럼들이 콤마로 분리된 텍스트 파일 포맷이다.

Numpy 라이브러리를 통해서 데이터를 읽어옴.
np.loadtxt("파일경로", 구분자, datatype)을 이용해서 파일을 읽고 data 변수에 array 형태로 넣어주게됨

파이썬 슬라이싱
기본 형태 a [start:end:step]
start : 슬라이싱을 시작할 시작위치
end : 슬라이싱을 끝낼 위치로 end는 포함하지 않음
step : stride 라고도 하며 몇개씩 끊어서 가져올지를 정함(옵션)

양수 혹은 음수 모두 가질 수 있음.
양수 : 연속적인 객체들의 앞에서부터 0을 시작으로 번호를 매김(ex 0, 1, 2, 3, 4)
음수 : 연속적인 객체들의 뒤에서부터 -1을 시작으로 번호를 매김(ex -5,-4,-3,-2,-1)

데이터의 크기가 너무 커 메모리가 감당하기 힘들 경우, 텐서플로우에서 지원하는 Queue Runners를 이용하여 해결할 수 있다.

batch : 나누어진 데이터 셋(data set)
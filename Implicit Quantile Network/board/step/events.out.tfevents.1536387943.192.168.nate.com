       £K"	  јўЏд÷Abrain.Event:2Е¶uhя     Ђ„£<	ZхЅўЏд÷A"џЊ
n
PlaceholderPlaceholder*
dtype0*'
_output_shapes
:€€€€€€€€€*
shape:€€€€€€€€€
p
Placeholder_1Placeholder*
dtype0*'
_output_shapes
:€€€€€€€€€*
shape:€€€€€€€€€
p
Placeholder_2Placeholder*
dtype0*'
_output_shapes
:€€€€€€€€€*
shape:€€€€€€€€€
p
Placeholder_3Placeholder*
dtype0*'
_output_shapes
:€€€€€€€€€*
shape:€€€€€€€€€
d
main/Tile/multiplesConst*
valueB"      *
dtype0*
_output_shapes
:
w
	main/TileTilePlaceholdermain/Tile/multiples*'
_output_shapes
:€€€€€€€€€ *

Tmultiples0*
T0
c
main/Reshape/shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:
v
main/ReshapeReshape	main/Tilemain/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
©
2main/dense/kernel/Initializer/random_uniform/shapeConst*$
_class
loc:@main/dense/kernel*
valueB"   А   *
dtype0*
_output_shapes
:
Ы
0main/dense/kernel/Initializer/random_uniform/minConst*$
_class
loc:@main/dense/kernel*
valueB
 *JQZЊ*
dtype0*
_output_shapes
: 
Ы
0main/dense/kernel/Initializer/random_uniform/maxConst*$
_class
loc:@main/dense/kernel*
valueB
 *JQZ>*
dtype0*
_output_shapes
: 
х
:main/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform2main/dense/kernel/Initializer/random_uniform/shape*

seed *
T0*$
_class
loc:@main/dense/kernel*
seed2 *
dtype0*
_output_shapes
:	А
в
0main/dense/kernel/Initializer/random_uniform/subSub0main/dense/kernel/Initializer/random_uniform/max0main/dense/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*$
_class
loc:@main/dense/kernel
х
0main/dense/kernel/Initializer/random_uniform/mulMul:main/dense/kernel/Initializer/random_uniform/RandomUniform0main/dense/kernel/Initializer/random_uniform/sub*
_output_shapes
:	А*
T0*$
_class
loc:@main/dense/kernel
з
,main/dense/kernel/Initializer/random_uniformAdd0main/dense/kernel/Initializer/random_uniform/mul0main/dense/kernel/Initializer/random_uniform/min*
_output_shapes
:	А*
T0*$
_class
loc:@main/dense/kernel
≠
main/dense/kernel
VariableV2*
shared_name *$
_class
loc:@main/dense/kernel*
	container *
shape:	А*
dtype0*
_output_shapes
:	А
№
main/dense/kernel/AssignAssignmain/dense/kernel,main/dense/kernel/Initializer/random_uniform*
validate_shape(*
_output_shapes
:	А*
use_locking(*
T0*$
_class
loc:@main/dense/kernel
Е
main/dense/kernel/readIdentitymain/dense/kernel*
T0*$
_class
loc:@main/dense/kernel*
_output_shapes
:	А
Ф
!main/dense/bias/Initializer/zerosConst*"
_class
loc:@main/dense/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
°
main/dense/bias
VariableV2*
	container *
shape:А*
dtype0*
_output_shapes	
:А*
shared_name *"
_class
loc:@main/dense/bias
«
main/dense/bias/AssignAssignmain/dense/bias!main/dense/bias/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@main/dense/bias*
validate_shape(*
_output_shapes	
:А
{
main/dense/bias/readIdentitymain/dense/bias*
T0*"
_class
loc:@main/dense/bias*
_output_shapes	
:А
Ъ
main/dense/MatMulMatMulmain/Reshapemain/dense/kernel/read*
transpose_b( *
T0*(
_output_shapes
:€€€€€€€€€А*
transpose_a( 
Р
main/dense/BiasAddBiasAddmain/dense/MatMulmain/dense/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:€€€€€€€€€А
^
main/dense/SeluSelumain/dense/BiasAdd*
T0*(
_output_shapes
:€€€€€€€€€А
e
main/Reshape_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"€€€€   
~
main/Reshape_1ReshapePlaceholder_1main/Reshape_1/shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
я

main/ConstConst*Ь
valueТBП@"А    џI@џ…@дЋAџIA—S{AдЋЦAянѓAџ…A÷1вA—SыAж:
BдЋBв\#Bян/BЁ~<BџIBЎ†UB÷1bB‘¬nB—S{BgтГBж:КBeГРBдЋЦBcЭBв\£B`•©BянѓB^6ґBЁ~ЉB\«¬Bџ…BYXѕBЎ†’BWйџB÷1вBUzиB‘¬оBRхB—SыB(ќ CgтCІCж:
C&_CeГC•ІCдЋC#рCcCҐ8 Cв\#C!Б&C`•)C†…,Cян/C3C^66CЮZ9CЁ~<C£?C\«BCЫлEC*
dtype0*
_output_shapes

:@
Й
main/MatMulMatMulmain/Reshape_1
main/Const*
T0*'
_output_shapes
:€€€€€€€€€@*
transpose_a( *
transpose_b( 
N
main/CosCosmain/MatMul*'
_output_shapes
:€€€€€€€€€@*
T0
≠
4main/dense_1/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*&
_class
loc:@main/dense_1/kernel*
valueB"@   А   
Я
2main/dense_1/kernel/Initializer/random_uniform/minConst*&
_class
loc:@main/dense_1/kernel*
valueB
 *у5Њ*
dtype0*
_output_shapes
: 
Я
2main/dense_1/kernel/Initializer/random_uniform/maxConst*&
_class
loc:@main/dense_1/kernel*
valueB
 *у5>*
dtype0*
_output_shapes
: 
ы
<main/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform4main/dense_1/kernel/Initializer/random_uniform/shape*

seed *
T0*&
_class
loc:@main/dense_1/kernel*
seed2 *
dtype0*
_output_shapes
:	@А
к
2main/dense_1/kernel/Initializer/random_uniform/subSub2main/dense_1/kernel/Initializer/random_uniform/max2main/dense_1/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@main/dense_1/kernel*
_output_shapes
: 
э
2main/dense_1/kernel/Initializer/random_uniform/mulMul<main/dense_1/kernel/Initializer/random_uniform/RandomUniform2main/dense_1/kernel/Initializer/random_uniform/sub*
_output_shapes
:	@А*
T0*&
_class
loc:@main/dense_1/kernel
п
.main/dense_1/kernel/Initializer/random_uniformAdd2main/dense_1/kernel/Initializer/random_uniform/mul2main/dense_1/kernel/Initializer/random_uniform/min*
_output_shapes
:	@А*
T0*&
_class
loc:@main/dense_1/kernel
±
main/dense_1/kernel
VariableV2*
shared_name *&
_class
loc:@main/dense_1/kernel*
	container *
shape:	@А*
dtype0*
_output_shapes
:	@А
д
main/dense_1/kernel/AssignAssignmain/dense_1/kernel.main/dense_1/kernel/Initializer/random_uniform*
T0*&
_class
loc:@main/dense_1/kernel*
validate_shape(*
_output_shapes
:	@А*
use_locking(
Л
main/dense_1/kernel/readIdentitymain/dense_1/kernel*
T0*&
_class
loc:@main/dense_1/kernel*
_output_shapes
:	@А
Ш
#main/dense_1/bias/Initializer/zerosConst*$
_class
loc:@main/dense_1/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
•
main/dense_1/bias
VariableV2*
shared_name *$
_class
loc:@main/dense_1/bias*
	container *
shape:А*
dtype0*
_output_shapes	
:А
ѕ
main/dense_1/bias/AssignAssignmain/dense_1/bias#main/dense_1/bias/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@main/dense_1/bias*
validate_shape(*
_output_shapes	
:А
Б
main/dense_1/bias/readIdentitymain/dense_1/bias*
T0*$
_class
loc:@main/dense_1/bias*
_output_shapes	
:А
Ъ
main/dense_1/MatMulMatMulmain/Cosmain/dense_1/kernel/read*
transpose_b( *
T0*(
_output_shapes
:€€€€€€€€€А*
transpose_a( 
Ц
main/dense_1/BiasAddBiasAddmain/dense_1/MatMulmain/dense_1/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:€€€€€€€€€А
b
main/dense_1/ReluRelumain/dense_1/BiasAdd*(
_output_shapes
:€€€€€€€€€А*
T0
f
main/MulMulmain/dense/Selumain/dense_1/Relu*
T0*(
_output_shapes
:€€€€€€€€€А
≠
4main/dense_2/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*&
_class
loc:@main/dense_2/kernel*
valueB"А      
Я
2main/dense_2/kernel/Initializer/random_uniform/minConst*&
_class
loc:@main/dense_2/kernel*
valueB
 *шK∆љ*
dtype0*
_output_shapes
: 
Я
2main/dense_2/kernel/Initializer/random_uniform/maxConst*&
_class
loc:@main/dense_2/kernel*
valueB
 *шK∆=*
dtype0*
_output_shapes
: 
ь
<main/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform4main/dense_2/kernel/Initializer/random_uniform/shape*
T0*&
_class
loc:@main/dense_2/kernel*
seed2 *
dtype0* 
_output_shapes
:
АА*

seed 
к
2main/dense_2/kernel/Initializer/random_uniform/subSub2main/dense_2/kernel/Initializer/random_uniform/max2main/dense_2/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*&
_class
loc:@main/dense_2/kernel
ю
2main/dense_2/kernel/Initializer/random_uniform/mulMul<main/dense_2/kernel/Initializer/random_uniform/RandomUniform2main/dense_2/kernel/Initializer/random_uniform/sub*
T0*&
_class
loc:@main/dense_2/kernel* 
_output_shapes
:
АА
р
.main/dense_2/kernel/Initializer/random_uniformAdd2main/dense_2/kernel/Initializer/random_uniform/mul2main/dense_2/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@main/dense_2/kernel* 
_output_shapes
:
АА
≥
main/dense_2/kernel
VariableV2*
	container *
shape:
АА*
dtype0* 
_output_shapes
:
АА*
shared_name *&
_class
loc:@main/dense_2/kernel
е
main/dense_2/kernel/AssignAssignmain/dense_2/kernel.main/dense_2/kernel/Initializer/random_uniform*
T0*&
_class
loc:@main/dense_2/kernel*
validate_shape(* 
_output_shapes
:
АА*
use_locking(
М
main/dense_2/kernel/readIdentitymain/dense_2/kernel*
T0*&
_class
loc:@main/dense_2/kernel* 
_output_shapes
:
АА
Ш
#main/dense_2/bias/Initializer/zerosConst*$
_class
loc:@main/dense_2/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
•
main/dense_2/bias
VariableV2*$
_class
loc:@main/dense_2/bias*
	container *
shape:А*
dtype0*
_output_shapes	
:А*
shared_name 
ѕ
main/dense_2/bias/AssignAssignmain/dense_2/bias#main/dense_2/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0*$
_class
loc:@main/dense_2/bias
Б
main/dense_2/bias/readIdentitymain/dense_2/bias*
T0*$
_class
loc:@main/dense_2/bias*
_output_shapes	
:А
Ъ
main/dense_2/MatMulMatMulmain/Mulmain/dense_2/kernel/read*
transpose_b( *
T0*(
_output_shapes
:€€€€€€€€€А*
transpose_a( 
Ц
main/dense_2/BiasAddBiasAddmain/dense_2/MatMulmain/dense_2/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:€€€€€€€€€А
b
main/dense_2/ReluRelumain/dense_2/BiasAdd*(
_output_shapes
:€€€€€€€€€А*
T0
≠
4main/dense_3/kernel/Initializer/random_uniform/shapeConst*&
_class
loc:@main/dense_3/kernel*
valueB"   А   *
dtype0*
_output_shapes
:
Я
2main/dense_3/kernel/Initializer/random_uniform/minConst*&
_class
loc:@main/dense_3/kernel*
valueB
 *шK∆љ*
dtype0*
_output_shapes
: 
Я
2main/dense_3/kernel/Initializer/random_uniform/maxConst*&
_class
loc:@main/dense_3/kernel*
valueB
 *шK∆=*
dtype0*
_output_shapes
: 
ь
<main/dense_3/kernel/Initializer/random_uniform/RandomUniformRandomUniform4main/dense_3/kernel/Initializer/random_uniform/shape*
T0*&
_class
loc:@main/dense_3/kernel*
seed2 *
dtype0* 
_output_shapes
:
АА*

seed 
к
2main/dense_3/kernel/Initializer/random_uniform/subSub2main/dense_3/kernel/Initializer/random_uniform/max2main/dense_3/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@main/dense_3/kernel*
_output_shapes
: 
ю
2main/dense_3/kernel/Initializer/random_uniform/mulMul<main/dense_3/kernel/Initializer/random_uniform/RandomUniform2main/dense_3/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
АА*
T0*&
_class
loc:@main/dense_3/kernel
р
.main/dense_3/kernel/Initializer/random_uniformAdd2main/dense_3/kernel/Initializer/random_uniform/mul2main/dense_3/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@main/dense_3/kernel* 
_output_shapes
:
АА
≥
main/dense_3/kernel
VariableV2*
shape:
АА*
dtype0* 
_output_shapes
:
АА*
shared_name *&
_class
loc:@main/dense_3/kernel*
	container 
е
main/dense_3/kernel/AssignAssignmain/dense_3/kernel.main/dense_3/kernel/Initializer/random_uniform*
validate_shape(* 
_output_shapes
:
АА*
use_locking(*
T0*&
_class
loc:@main/dense_3/kernel
М
main/dense_3/kernel/readIdentitymain/dense_3/kernel* 
_output_shapes
:
АА*
T0*&
_class
loc:@main/dense_3/kernel
Ш
#main/dense_3/bias/Initializer/zerosConst*$
_class
loc:@main/dense_3/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
•
main/dense_3/bias
VariableV2*
dtype0*
_output_shapes	
:А*
shared_name *$
_class
loc:@main/dense_3/bias*
	container *
shape:А
ѕ
main/dense_3/bias/AssignAssignmain/dense_3/bias#main/dense_3/bias/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@main/dense_3/bias*
validate_shape(*
_output_shapes	
:А
Б
main/dense_3/bias/readIdentitymain/dense_3/bias*
T0*$
_class
loc:@main/dense_3/bias*
_output_shapes	
:А
£
main/dense_3/MatMulMatMulmain/dense_2/Relumain/dense_3/kernel/read*
T0*(
_output_shapes
:€€€€€€€€€А*
transpose_a( *
transpose_b( 
Ц
main/dense_3/BiasAddBiasAddmain/dense_3/MatMulmain/dense_3/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:€€€€€€€€€А
b
main/dense_3/ReluRelumain/dense_3/BiasAdd*
T0*(
_output_shapes
:€€€€€€€€€А
≠
4main/dense_4/kernel/Initializer/random_uniform/shapeConst*&
_class
loc:@main/dense_4/kernel*
valueB"А      *
dtype0*
_output_shapes
:
Я
2main/dense_4/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *&
_class
loc:@main/dense_4/kernel*
valueB
 *Сэ[Њ
Я
2main/dense_4/kernel/Initializer/random_uniform/maxConst*&
_class
loc:@main/dense_4/kernel*
valueB
 *Сэ[>*
dtype0*
_output_shapes
: 
ы
<main/dense_4/kernel/Initializer/random_uniform/RandomUniformRandomUniform4main/dense_4/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	А*

seed *
T0*&
_class
loc:@main/dense_4/kernel*
seed2 
к
2main/dense_4/kernel/Initializer/random_uniform/subSub2main/dense_4/kernel/Initializer/random_uniform/max2main/dense_4/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@main/dense_4/kernel*
_output_shapes
: 
э
2main/dense_4/kernel/Initializer/random_uniform/mulMul<main/dense_4/kernel/Initializer/random_uniform/RandomUniform2main/dense_4/kernel/Initializer/random_uniform/sub*
_output_shapes
:	А*
T0*&
_class
loc:@main/dense_4/kernel
п
.main/dense_4/kernel/Initializer/random_uniformAdd2main/dense_4/kernel/Initializer/random_uniform/mul2main/dense_4/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@main/dense_4/kernel*
_output_shapes
:	А
±
main/dense_4/kernel
VariableV2*
dtype0*
_output_shapes
:	А*
shared_name *&
_class
loc:@main/dense_4/kernel*
	container *
shape:	А
д
main/dense_4/kernel/AssignAssignmain/dense_4/kernel.main/dense_4/kernel/Initializer/random_uniform*
T0*&
_class
loc:@main/dense_4/kernel*
validate_shape(*
_output_shapes
:	А*
use_locking(
Л
main/dense_4/kernel/readIdentitymain/dense_4/kernel*
T0*&
_class
loc:@main/dense_4/kernel*
_output_shapes
:	А
Ц
#main/dense_4/bias/Initializer/zerosConst*$
_class
loc:@main/dense_4/bias*
valueB*    *
dtype0*
_output_shapes
:
£
main/dense_4/bias
VariableV2*
shared_name *$
_class
loc:@main/dense_4/bias*
	container *
shape:*
dtype0*
_output_shapes
:
ќ
main/dense_4/bias/AssignAssignmain/dense_4/bias#main/dense_4/bias/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@main/dense_4/bias*
validate_shape(*
_output_shapes
:
А
main/dense_4/bias/readIdentitymain/dense_4/bias*
T0*$
_class
loc:@main/dense_4/bias*
_output_shapes
:
Ґ
main/dense_4/MatMulMatMulmain/dense_3/Relumain/dense_4/kernel/read*
T0*'
_output_shapes
:€€€€€€€€€*
transpose_a( *
transpose_b( 
Х
main/dense_4/BiasAddBiasAddmain/dense_4/MatMulmain/dense_4/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€
N
main/Const_1Const*
value	B : *
dtype0*
_output_shapes
: 
V
main/split/split_dimConst*
value	B : *
dtype0*
_output_shapes
: 
“

main/splitSplitmain/split/split_dimmain/dense_4/BiasAdd*
T0*ц
_output_shapesг
а:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€*
	num_split 
Ј
main/transpose/xPack
main/splitmain/split:1main/split:2main/split:3main/split:4main/split:5main/split:6main/split:7main/split:8main/split:9main/split:10main/split:11main/split:12main/split:13main/split:14main/split:15main/split:16main/split:17main/split:18main/split:19main/split:20main/split:21main/split:22main/split:23main/split:24main/split:25main/split:26main/split:27main/split:28main/split:29main/split:30main/split:31*
T0*

axis *
N *+
_output_shapes
: €€€€€€€€€
h
main/transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:
Е
main/transpose	Transposemain/transpose/xmain/transpose/perm*+
_output_shapes
: €€€€€€€€€*
Tperm0*
T0
f
target/Tile/multiplesConst*
dtype0*
_output_shapes
:*
valueB"      
{
target/TileTilePlaceholdertarget/Tile/multiples*'
_output_shapes
:€€€€€€€€€ *

Tmultiples0*
T0
e
target/Reshape/shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:
|
target/ReshapeReshapetarget/Tiletarget/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
≠
4target/dense/kernel/Initializer/random_uniform/shapeConst*&
_class
loc:@target/dense/kernel*
valueB"   А   *
dtype0*
_output_shapes
:
Я
2target/dense/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *&
_class
loc:@target/dense/kernel*
valueB
 *JQZЊ
Я
2target/dense/kernel/Initializer/random_uniform/maxConst*&
_class
loc:@target/dense/kernel*
valueB
 *JQZ>*
dtype0*
_output_shapes
: 
ы
<target/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform4target/dense/kernel/Initializer/random_uniform/shape*
seed2 *
dtype0*
_output_shapes
:	А*

seed *
T0*&
_class
loc:@target/dense/kernel
к
2target/dense/kernel/Initializer/random_uniform/subSub2target/dense/kernel/Initializer/random_uniform/max2target/dense/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@target/dense/kernel*
_output_shapes
: 
э
2target/dense/kernel/Initializer/random_uniform/mulMul<target/dense/kernel/Initializer/random_uniform/RandomUniform2target/dense/kernel/Initializer/random_uniform/sub*
T0*&
_class
loc:@target/dense/kernel*
_output_shapes
:	А
п
.target/dense/kernel/Initializer/random_uniformAdd2target/dense/kernel/Initializer/random_uniform/mul2target/dense/kernel/Initializer/random_uniform/min*
_output_shapes
:	А*
T0*&
_class
loc:@target/dense/kernel
±
target/dense/kernel
VariableV2*
shared_name *&
_class
loc:@target/dense/kernel*
	container *
shape:	А*
dtype0*
_output_shapes
:	А
д
target/dense/kernel/AssignAssigntarget/dense/kernel.target/dense/kernel/Initializer/random_uniform*
validate_shape(*
_output_shapes
:	А*
use_locking(*
T0*&
_class
loc:@target/dense/kernel
Л
target/dense/kernel/readIdentitytarget/dense/kernel*
_output_shapes
:	А*
T0*&
_class
loc:@target/dense/kernel
Ш
#target/dense/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:А*$
_class
loc:@target/dense/bias*
valueBА*    
•
target/dense/bias
VariableV2*
dtype0*
_output_shapes	
:А*
shared_name *$
_class
loc:@target/dense/bias*
	container *
shape:А
ѕ
target/dense/bias/AssignAssigntarget/dense/bias#target/dense/bias/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@target/dense/bias*
validate_shape(*
_output_shapes	
:А
Б
target/dense/bias/readIdentitytarget/dense/bias*
T0*$
_class
loc:@target/dense/bias*
_output_shapes	
:А
†
target/dense/MatMulMatMultarget/Reshapetarget/dense/kernel/read*(
_output_shapes
:€€€€€€€€€А*
transpose_a( *
transpose_b( *
T0
Ц
target/dense/BiasAddBiasAddtarget/dense/MatMultarget/dense/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:€€€€€€€€€А
b
target/dense/SeluSelutarget/dense/BiasAdd*
T0*(
_output_shapes
:€€€€€€€€€А
g
target/Reshape_1/shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:
В
target/Reshape_1ReshapePlaceholder_1target/Reshape_1/shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
б
target/ConstConst*Ь
valueТBП@"А    џI@џ…@дЋAџIA—S{AдЋЦAянѓAџ…A÷1вA—SыAж:
BдЋBв\#Bян/BЁ~<BџIBЎ†UB÷1bB‘¬nB—S{BgтГBж:КBeГРBдЋЦBcЭBв\£B`•©BянѓB^6ґBЁ~ЉB\«¬Bџ…BYXѕBЎ†’BWйџB÷1вBUzиB‘¬оBRхB—SыB(ќ CgтCІCж:
C&_CeГC•ІCдЋC#рCcCҐ8 Cв\#C!Б&C`•)C†…,Cян/C3C^66CЮZ9CЁ~<C£?C\«BCЫлEC*
dtype0*
_output_shapes

:@
П
target/MatMulMatMultarget/Reshape_1target/Const*'
_output_shapes
:€€€€€€€€€@*
transpose_a( *
transpose_b( *
T0
R

target/CosCostarget/MatMul*
T0*'
_output_shapes
:€€€€€€€€€@
±
6target/dense_1/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*(
_class
loc:@target/dense_1/kernel*
valueB"@   А   
£
4target/dense_1/kernel/Initializer/random_uniform/minConst*(
_class
loc:@target/dense_1/kernel*
valueB
 *у5Њ*
dtype0*
_output_shapes
: 
£
4target/dense_1/kernel/Initializer/random_uniform/maxConst*(
_class
loc:@target/dense_1/kernel*
valueB
 *у5>*
dtype0*
_output_shapes
: 
Б
>target/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform6target/dense_1/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	@А*

seed *
T0*(
_class
loc:@target/dense_1/kernel*
seed2 
т
4target/dense_1/kernel/Initializer/random_uniform/subSub4target/dense_1/kernel/Initializer/random_uniform/max4target/dense_1/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*(
_class
loc:@target/dense_1/kernel
Е
4target/dense_1/kernel/Initializer/random_uniform/mulMul>target/dense_1/kernel/Initializer/random_uniform/RandomUniform4target/dense_1/kernel/Initializer/random_uniform/sub*
T0*(
_class
loc:@target/dense_1/kernel*
_output_shapes
:	@А
ч
0target/dense_1/kernel/Initializer/random_uniformAdd4target/dense_1/kernel/Initializer/random_uniform/mul4target/dense_1/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@target/dense_1/kernel*
_output_shapes
:	@А
µ
target/dense_1/kernel
VariableV2*(
_class
loc:@target/dense_1/kernel*
	container *
shape:	@А*
dtype0*
_output_shapes
:	@А*
shared_name 
м
target/dense_1/kernel/AssignAssigntarget/dense_1/kernel0target/dense_1/kernel/Initializer/random_uniform*
validate_shape(*
_output_shapes
:	@А*
use_locking(*
T0*(
_class
loc:@target/dense_1/kernel
С
target/dense_1/kernel/readIdentitytarget/dense_1/kernel*
T0*(
_class
loc:@target/dense_1/kernel*
_output_shapes
:	@А
Ь
%target/dense_1/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:А*&
_class
loc:@target/dense_1/bias*
valueBА*    
©
target/dense_1/bias
VariableV2*
dtype0*
_output_shapes	
:А*
shared_name *&
_class
loc:@target/dense_1/bias*
	container *
shape:А
„
target/dense_1/bias/AssignAssigntarget/dense_1/bias%target/dense_1/bias/Initializer/zeros*
use_locking(*
T0*&
_class
loc:@target/dense_1/bias*
validate_shape(*
_output_shapes	
:А
З
target/dense_1/bias/readIdentitytarget/dense_1/bias*
T0*&
_class
loc:@target/dense_1/bias*
_output_shapes	
:А
†
target/dense_1/MatMulMatMul
target/Costarget/dense_1/kernel/read*(
_output_shapes
:€€€€€€€€€А*
transpose_a( *
transpose_b( *
T0
Ь
target/dense_1/BiasAddBiasAddtarget/dense_1/MatMultarget/dense_1/bias/read*
data_formatNHWC*(
_output_shapes
:€€€€€€€€€А*
T0
f
target/dense_1/ReluRelutarget/dense_1/BiasAdd*(
_output_shapes
:€€€€€€€€€А*
T0
l

target/MulMultarget/dense/Selutarget/dense_1/Relu*(
_output_shapes
:€€€€€€€€€А*
T0
±
6target/dense_2/kernel/Initializer/random_uniform/shapeConst*(
_class
loc:@target/dense_2/kernel*
valueB"А      *
dtype0*
_output_shapes
:
£
4target/dense_2/kernel/Initializer/random_uniform/minConst*(
_class
loc:@target/dense_2/kernel*
valueB
 *шK∆љ*
dtype0*
_output_shapes
: 
£
4target/dense_2/kernel/Initializer/random_uniform/maxConst*(
_class
loc:@target/dense_2/kernel*
valueB
 *шK∆=*
dtype0*
_output_shapes
: 
В
>target/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform6target/dense_2/kernel/Initializer/random_uniform/shape*
T0*(
_class
loc:@target/dense_2/kernel*
seed2 *
dtype0* 
_output_shapes
:
АА*

seed 
т
4target/dense_2/kernel/Initializer/random_uniform/subSub4target/dense_2/kernel/Initializer/random_uniform/max4target/dense_2/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@target/dense_2/kernel*
_output_shapes
: 
Ж
4target/dense_2/kernel/Initializer/random_uniform/mulMul>target/dense_2/kernel/Initializer/random_uniform/RandomUniform4target/dense_2/kernel/Initializer/random_uniform/sub*
T0*(
_class
loc:@target/dense_2/kernel* 
_output_shapes
:
АА
ш
0target/dense_2/kernel/Initializer/random_uniformAdd4target/dense_2/kernel/Initializer/random_uniform/mul4target/dense_2/kernel/Initializer/random_uniform/min* 
_output_shapes
:
АА*
T0*(
_class
loc:@target/dense_2/kernel
Ј
target/dense_2/kernel
VariableV2*
dtype0* 
_output_shapes
:
АА*
shared_name *(
_class
loc:@target/dense_2/kernel*
	container *
shape:
АА
н
target/dense_2/kernel/AssignAssigntarget/dense_2/kernel0target/dense_2/kernel/Initializer/random_uniform*
T0*(
_class
loc:@target/dense_2/kernel*
validate_shape(* 
_output_shapes
:
АА*
use_locking(
Т
target/dense_2/kernel/readIdentitytarget/dense_2/kernel* 
_output_shapes
:
АА*
T0*(
_class
loc:@target/dense_2/kernel
Ь
%target/dense_2/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:А*&
_class
loc:@target/dense_2/bias*
valueBА*    
©
target/dense_2/bias
VariableV2*
	container *
shape:А*
dtype0*
_output_shapes	
:А*
shared_name *&
_class
loc:@target/dense_2/bias
„
target/dense_2/bias/AssignAssigntarget/dense_2/bias%target/dense_2/bias/Initializer/zeros*
use_locking(*
T0*&
_class
loc:@target/dense_2/bias*
validate_shape(*
_output_shapes	
:А
З
target/dense_2/bias/readIdentitytarget/dense_2/bias*
T0*&
_class
loc:@target/dense_2/bias*
_output_shapes	
:А
†
target/dense_2/MatMulMatMul
target/Multarget/dense_2/kernel/read*
T0*(
_output_shapes
:€€€€€€€€€А*
transpose_a( *
transpose_b( 
Ь
target/dense_2/BiasAddBiasAddtarget/dense_2/MatMultarget/dense_2/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:€€€€€€€€€А
f
target/dense_2/ReluRelutarget/dense_2/BiasAdd*
T0*(
_output_shapes
:€€€€€€€€€А
±
6target/dense_3/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*(
_class
loc:@target/dense_3/kernel*
valueB"   А   
£
4target/dense_3/kernel/Initializer/random_uniform/minConst*(
_class
loc:@target/dense_3/kernel*
valueB
 *шK∆љ*
dtype0*
_output_shapes
: 
£
4target/dense_3/kernel/Initializer/random_uniform/maxConst*(
_class
loc:@target/dense_3/kernel*
valueB
 *шK∆=*
dtype0*
_output_shapes
: 
В
>target/dense_3/kernel/Initializer/random_uniform/RandomUniformRandomUniform6target/dense_3/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
АА*

seed *
T0*(
_class
loc:@target/dense_3/kernel*
seed2 
т
4target/dense_3/kernel/Initializer/random_uniform/subSub4target/dense_3/kernel/Initializer/random_uniform/max4target/dense_3/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@target/dense_3/kernel*
_output_shapes
: 
Ж
4target/dense_3/kernel/Initializer/random_uniform/mulMul>target/dense_3/kernel/Initializer/random_uniform/RandomUniform4target/dense_3/kernel/Initializer/random_uniform/sub*
T0*(
_class
loc:@target/dense_3/kernel* 
_output_shapes
:
АА
ш
0target/dense_3/kernel/Initializer/random_uniformAdd4target/dense_3/kernel/Initializer/random_uniform/mul4target/dense_3/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@target/dense_3/kernel* 
_output_shapes
:
АА
Ј
target/dense_3/kernel
VariableV2*(
_class
loc:@target/dense_3/kernel*
	container *
shape:
АА*
dtype0* 
_output_shapes
:
АА*
shared_name 
н
target/dense_3/kernel/AssignAssigntarget/dense_3/kernel0target/dense_3/kernel/Initializer/random_uniform*
T0*(
_class
loc:@target/dense_3/kernel*
validate_shape(* 
_output_shapes
:
АА*
use_locking(
Т
target/dense_3/kernel/readIdentitytarget/dense_3/kernel*
T0*(
_class
loc:@target/dense_3/kernel* 
_output_shapes
:
АА
Ь
%target/dense_3/bias/Initializer/zerosConst*&
_class
loc:@target/dense_3/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
©
target/dense_3/bias
VariableV2*
shared_name *&
_class
loc:@target/dense_3/bias*
	container *
shape:А*
dtype0*
_output_shapes	
:А
„
target/dense_3/bias/AssignAssigntarget/dense_3/bias%target/dense_3/bias/Initializer/zeros*
use_locking(*
T0*&
_class
loc:@target/dense_3/bias*
validate_shape(*
_output_shapes	
:А
З
target/dense_3/bias/readIdentitytarget/dense_3/bias*
_output_shapes	
:А*
T0*&
_class
loc:@target/dense_3/bias
©
target/dense_3/MatMulMatMultarget/dense_2/Relutarget/dense_3/kernel/read*
T0*(
_output_shapes
:€€€€€€€€€А*
transpose_a( *
transpose_b( 
Ь
target/dense_3/BiasAddBiasAddtarget/dense_3/MatMultarget/dense_3/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:€€€€€€€€€А
f
target/dense_3/ReluRelutarget/dense_3/BiasAdd*(
_output_shapes
:€€€€€€€€€А*
T0
±
6target/dense_4/kernel/Initializer/random_uniform/shapeConst*(
_class
loc:@target/dense_4/kernel*
valueB"А      *
dtype0*
_output_shapes
:
£
4target/dense_4/kernel/Initializer/random_uniform/minConst*(
_class
loc:@target/dense_4/kernel*
valueB
 *Сэ[Њ*
dtype0*
_output_shapes
: 
£
4target/dense_4/kernel/Initializer/random_uniform/maxConst*(
_class
loc:@target/dense_4/kernel*
valueB
 *Сэ[>*
dtype0*
_output_shapes
: 
Б
>target/dense_4/kernel/Initializer/random_uniform/RandomUniformRandomUniform6target/dense_4/kernel/Initializer/random_uniform/shape*
seed2 *
dtype0*
_output_shapes
:	А*

seed *
T0*(
_class
loc:@target/dense_4/kernel
т
4target/dense_4/kernel/Initializer/random_uniform/subSub4target/dense_4/kernel/Initializer/random_uniform/max4target/dense_4/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@target/dense_4/kernel*
_output_shapes
: 
Е
4target/dense_4/kernel/Initializer/random_uniform/mulMul>target/dense_4/kernel/Initializer/random_uniform/RandomUniform4target/dense_4/kernel/Initializer/random_uniform/sub*
T0*(
_class
loc:@target/dense_4/kernel*
_output_shapes
:	А
ч
0target/dense_4/kernel/Initializer/random_uniformAdd4target/dense_4/kernel/Initializer/random_uniform/mul4target/dense_4/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@target/dense_4/kernel*
_output_shapes
:	А
µ
target/dense_4/kernel
VariableV2*
shared_name *(
_class
loc:@target/dense_4/kernel*
	container *
shape:	А*
dtype0*
_output_shapes
:	А
м
target/dense_4/kernel/AssignAssigntarget/dense_4/kernel0target/dense_4/kernel/Initializer/random_uniform*
T0*(
_class
loc:@target/dense_4/kernel*
validate_shape(*
_output_shapes
:	А*
use_locking(
С
target/dense_4/kernel/readIdentitytarget/dense_4/kernel*
T0*(
_class
loc:@target/dense_4/kernel*
_output_shapes
:	А
Ъ
%target/dense_4/bias/Initializer/zerosConst*&
_class
loc:@target/dense_4/bias*
valueB*    *
dtype0*
_output_shapes
:
І
target/dense_4/bias
VariableV2*&
_class
loc:@target/dense_4/bias*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
÷
target/dense_4/bias/AssignAssigntarget/dense_4/bias%target/dense_4/bias/Initializer/zeros*
use_locking(*
T0*&
_class
loc:@target/dense_4/bias*
validate_shape(*
_output_shapes
:
Ж
target/dense_4/bias/readIdentitytarget/dense_4/bias*
T0*&
_class
loc:@target/dense_4/bias*
_output_shapes
:
®
target/dense_4/MatMulMatMultarget/dense_3/Relutarget/dense_4/kernel/read*
T0*'
_output_shapes
:€€€€€€€€€*
transpose_a( *
transpose_b( 
Ы
target/dense_4/BiasAddBiasAddtarget/dense_4/MatMultarget/dense_4/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€
P
target/Const_1Const*
dtype0*
_output_shapes
: *
value	B : 
X
target/split/split_dimConst*
value	B : *
dtype0*
_output_shapes
: 
Ў
target/splitSplittarget/split/split_dimtarget/dense_4/BiasAdd*
T0*ц
_output_shapesг
а:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€*
	num_split 
щ
target/transpose/xPacktarget/splittarget/split:1target/split:2target/split:3target/split:4target/split:5target/split:6target/split:7target/split:8target/split:9target/split:10target/split:11target/split:12target/split:13target/split:14target/split:15target/split:16target/split:17target/split:18target/split:19target/split:20target/split:21target/split:22target/split:23target/split:24target/split:25target/split:26target/split:27target/split:28target/split:29target/split:30target/split:31*
N *+
_output_shapes
: €€€€€€€€€*
T0*

axis 
j
target/transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:
Л
target/transpose	Transposetarget/transpose/xtarget/transpose/perm*+
_output_shapes
: €€€€€€€€€*
Tperm0*
T0
Y
ExpandDims/dimConst*
dtype0*
_output_shapes
: *
valueB :
€€€€€€€€€
y

ExpandDims
ExpandDimsPlaceholder_3ExpandDims/dim*+
_output_shapes
:€€€€€€€€€*

Tdim0*
T0
\
mulMulmain/transpose
ExpandDims*
T0*+
_output_shapes
: €€€€€€€€€
W
Sum/reduction_indicesConst*
dtype0*
_output_shapes
: *
value	B :
u
SumSummulSum/reduction_indices*
T0*'
_output_shapes
: €€€€€€€€€*
	keep_dims( *

Tidx0
R
huber_loss/SubSubSumPlaceholder_2*
T0*
_output_shapes

: 
N
huber_loss/AbsAbshuber_loss/Sub*
_output_shapes

: *
T0
Y
huber_loss/Minimum/yConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
l
huber_loss/MinimumMinimumhuber_loss/Abshuber_loss/Minimum/y*
T0*
_output_shapes

: 
d
huber_loss/Sub_1Subhuber_loss/Abshuber_loss/Minimum*
T0*
_output_shapes

: 
U
huber_loss/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *   ?
f
huber_loss/MulMulhuber_loss/Minimumhuber_loss/Minimum*
_output_shapes

: *
T0
b
huber_loss/Mul_1Mulhuber_loss/Consthuber_loss/Mul*
T0*
_output_shapes

: 
W
huber_loss/Mul_2/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
f
huber_loss/Mul_2Mulhuber_loss/Mul_2/xhuber_loss/Sub_1*
T0*
_output_shapes

: 
b
huber_loss/AddAddhuber_loss/Mul_1huber_loss/Mul_2*
T0*
_output_shapes

: 
l
'huber_loss/assert_broadcastable/weightsConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
p
-huber_loss/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 
n
,huber_loss/assert_broadcastable/weights/rankConst*
value	B : *
dtype0*
_output_shapes
: 
}
,huber_loss/assert_broadcastable/values/shapeConst*
valueB"       *
dtype0*
_output_shapes
:
m
+huber_loss/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
C
;huber_loss/assert_broadcastable/static_scalar_check_successNoOp
Щ
huber_loss/ToFloat_3/xConst<^huber_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *  А?*
dtype0*
_output_shapes
: 
h
huber_loss/Mul_3Mulhuber_loss/Addhuber_loss/ToFloat_3/x*
T0*
_output_shapes

: 
J
sub/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
R
subSubsub/xPlaceholder_1*
T0*'
_output_shapes
:€€€€€€€€€
I
sub_1SubPlaceholder_2Sum*
T0*
_output_shapes

: 
K
Less/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
D
LessLesssub_1Less/y*
T0*
_output_shapes

: 
L
mul_1Mulsubhuber_loss/Mul_3*
T0*
_output_shapes

: 
V
mul_2MulPlaceholder_1huber_loss/Mul_3*
T0*
_output_shapes

: 
M
SelectSelectLessmul_1mul_2*
_output_shapes

: *
T0
Y
Sum_1/reduction_indicesConst*
dtype0*
_output_shapes
: *
value	B :
o
Sum_1SumSelectSum_1/reduction_indices*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
O
ConstConst*
valueB: *
dtype0*
_output_shapes
:
X
MeanMeanSum_1Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
X
gradients/grad_ys_0Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
k
!gradients/Mean_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
М
gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
c
gradients/Mean_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
П
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Const*
T0*
_output_shapes
: *

Tmultiples0
`
gradients/Mean_grad/Const_1Const*
valueB
 *   B*
dtype0*
_output_shapes
: 
В
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Const_1*
T0*
_output_shapes
: 
k
gradients/Sum_1_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"       
К
gradients/Sum_1_grad/SizeConst*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
£
gradients/Sum_1_grad/addAddSum_1/reduction_indicesgradients/Sum_1_grad/Size*
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
_output_shapes
: 
©
gradients/Sum_1_grad/modFloorModgradients/Sum_1_grad/addgradients/Sum_1_grad/Size*
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
_output_shapes
: 
О
gradients/Sum_1_grad/Shape_1Const*
dtype0*
_output_shapes
: *-
_class#
!loc:@gradients/Sum_1_grad/Shape*
valueB 
С
 gradients/Sum_1_grad/range/startConst*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
value	B : *
dtype0*
_output_shapes
: 
С
 gradients/Sum_1_grad/range/deltaConst*
dtype0*
_output_shapes
: *-
_class#
!loc:@gradients/Sum_1_grad/Shape*
value	B :
ў
gradients/Sum_1_grad/rangeRange gradients/Sum_1_grad/range/startgradients/Sum_1_grad/Size gradients/Sum_1_grad/range/delta*

Tidx0*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
_output_shapes
:
Р
gradients/Sum_1_grad/Fill/valueConst*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
¬
gradients/Sum_1_grad/FillFillgradients/Sum_1_grad/Shape_1gradients/Sum_1_grad/Fill/value*
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape*

index_type0*
_output_shapes
: 
Ж
"gradients/Sum_1_grad/DynamicStitchDynamicStitchgradients/Sum_1_grad/rangegradients/Sum_1_grad/modgradients/Sum_1_grad/Shapegradients/Sum_1_grad/Fill*
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
N*#
_output_shapes
:€€€€€€€€€
П
gradients/Sum_1_grad/Maximum/yConst*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
»
gradients/Sum_1_grad/MaximumMaximum"gradients/Sum_1_grad/DynamicStitchgradients/Sum_1_grad/Maximum/y*
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape*#
_output_shapes
:€€€€€€€€€
Ј
gradients/Sum_1_grad/floordivFloorDivgradients/Sum_1_grad/Shapegradients/Sum_1_grad/Maximum*
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
_output_shapes
:
Щ
gradients/Sum_1_grad/ReshapeReshapegradients/Mean_grad/truediv"gradients/Sum_1_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
Щ
gradients/Sum_1_grad/TileTilegradients/Sum_1_grad/Reshapegradients/Sum_1_grad/floordiv*
T0*
_output_shapes

: *

Tmultiples0
u
 gradients/Select_grad/zeros_likeConst*
dtype0*
_output_shapes

: *
valueB *    
Т
gradients/Select_grad/SelectSelectLessgradients/Sum_1_grad/Tile gradients/Select_grad/zeros_like*
_output_shapes

: *
T0
Ф
gradients/Select_grad/Select_1SelectLess gradients/Select_grad/zeros_likegradients/Sum_1_grad/Tile*
T0*
_output_shapes

: 
n
&gradients/Select_grad/tuple/group_depsNoOp^gradients/Select_grad/Select^gradients/Select_grad/Select_1
џ
.gradients/Select_grad/tuple/control_dependencyIdentitygradients/Select_grad/Select'^gradients/Select_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/Select_grad/Select*
_output_shapes

: 
б
0gradients/Select_grad/tuple/control_dependency_1Identitygradients/Select_grad/Select_1'^gradients/Select_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/Select_grad/Select_1*
_output_shapes

: 
]
gradients/mul_1_grad/ShapeShapesub*
_output_shapes
:*
T0*
out_type0
m
gradients/mul_1_grad/Shape_1Const*
valueB"       *
dtype0*
_output_shapes
:
Ї
*gradients/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_1_grad/Shapegradients/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
К
gradients/mul_1_grad/MulMul.gradients/Select_grad/tuple/control_dependencyhuber_loss/Mul_3*
T0*
_output_shapes

: 
•
gradients/mul_1_grad/SumSumgradients/mul_1_grad/Mul*gradients/mul_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Э
gradients/mul_1_grad/ReshapeReshapegradients/mul_1_grad/Sumgradients/mul_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€

gradients/mul_1_grad/Mul_1Mulsub.gradients/Select_grad/tuple/control_dependency*
T0*
_output_shapes

: 
Ђ
gradients/mul_1_grad/Sum_1Sumgradients/mul_1_grad/Mul_1,gradients/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ъ
gradients/mul_1_grad/Reshape_1Reshapegradients/mul_1_grad/Sum_1gradients/mul_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes

: 
m
%gradients/mul_1_grad/tuple/group_depsNoOp^gradients/mul_1_grad/Reshape^gradients/mul_1_grad/Reshape_1
в
-gradients/mul_1_grad/tuple/control_dependencyIdentitygradients/mul_1_grad/Reshape&^gradients/mul_1_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/mul_1_grad/Reshape*'
_output_shapes
:€€€€€€€€€
я
/gradients/mul_1_grad/tuple/control_dependency_1Identitygradients/mul_1_grad/Reshape_1&^gradients/mul_1_grad/tuple/group_deps*
_output_shapes

: *
T0*1
_class'
%#loc:@gradients/mul_1_grad/Reshape_1
g
gradients/mul_2_grad/ShapeShapePlaceholder_1*
_output_shapes
:*
T0*
out_type0
m
gradients/mul_2_grad/Shape_1Const*
valueB"       *
dtype0*
_output_shapes
:
Ї
*gradients/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_2_grad/Shapegradients/mul_2_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
М
gradients/mul_2_grad/MulMul0gradients/Select_grad/tuple/control_dependency_1huber_loss/Mul_3*
T0*
_output_shapes

: 
•
gradients/mul_2_grad/SumSumgradients/mul_2_grad/Mul*gradients/mul_2_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Э
gradients/mul_2_grad/ReshapeReshapegradients/mul_2_grad/Sumgradients/mul_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
Л
gradients/mul_2_grad/Mul_1MulPlaceholder_10gradients/Select_grad/tuple/control_dependency_1*
_output_shapes

: *
T0
Ђ
gradients/mul_2_grad/Sum_1Sumgradients/mul_2_grad/Mul_1,gradients/mul_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ъ
gradients/mul_2_grad/Reshape_1Reshapegradients/mul_2_grad/Sum_1gradients/mul_2_grad/Shape_1*
_output_shapes

: *
T0*
Tshape0
m
%gradients/mul_2_grad/tuple/group_depsNoOp^gradients/mul_2_grad/Reshape^gradients/mul_2_grad/Reshape_1
в
-gradients/mul_2_grad/tuple/control_dependencyIdentitygradients/mul_2_grad/Reshape&^gradients/mul_2_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/mul_2_grad/Reshape*'
_output_shapes
:€€€€€€€€€
я
/gradients/mul_2_grad/tuple/control_dependency_1Identitygradients/mul_2_grad/Reshape_1&^gradients/mul_2_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/mul_2_grad/Reshape_1*
_output_shapes

: 
Ё
gradients/AddNAddN/gradients/mul_1_grad/tuple/control_dependency_1/gradients/mul_2_grad/tuple/control_dependency_1*
T0*1
_class'
%#loc:@gradients/mul_1_grad/Reshape_1*
N*
_output_shapes

: 
v
%gradients/huber_loss/Mul_3_grad/ShapeConst*
valueB"       *
dtype0*
_output_shapes
:
j
'gradients/huber_loss/Mul_3_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
џ
5gradients/huber_loss/Mul_3_grad/BroadcastGradientArgsBroadcastGradientArgs%gradients/huber_loss/Mul_3_grad/Shape'gradients/huber_loss/Mul_3_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
{
#gradients/huber_loss/Mul_3_grad/MulMulgradients/AddNhuber_loss/ToFloat_3/x*
T0*
_output_shapes

: 
∆
#gradients/huber_loss/Mul_3_grad/SumSum#gradients/huber_loss/Mul_3_grad/Mul5gradients/huber_loss/Mul_3_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
µ
'gradients/huber_loss/Mul_3_grad/ReshapeReshape#gradients/huber_loss/Mul_3_grad/Sum%gradients/huber_loss/Mul_3_grad/Shape*
T0*
Tshape0*
_output_shapes

: 
u
%gradients/huber_loss/Mul_3_grad/Mul_1Mulhuber_loss/Addgradients/AddN*
_output_shapes

: *
T0
ћ
%gradients/huber_loss/Mul_3_grad/Sum_1Sum%gradients/huber_loss/Mul_3_grad/Mul_17gradients/huber_loss/Mul_3_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
≥
)gradients/huber_loss/Mul_3_grad/Reshape_1Reshape%gradients/huber_loss/Mul_3_grad/Sum_1'gradients/huber_loss/Mul_3_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
О
0gradients/huber_loss/Mul_3_grad/tuple/group_depsNoOp(^gradients/huber_loss/Mul_3_grad/Reshape*^gradients/huber_loss/Mul_3_grad/Reshape_1
Е
8gradients/huber_loss/Mul_3_grad/tuple/control_dependencyIdentity'gradients/huber_loss/Mul_3_grad/Reshape1^gradients/huber_loss/Mul_3_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/huber_loss/Mul_3_grad/Reshape*
_output_shapes

: 
Г
:gradients/huber_loss/Mul_3_grad/tuple/control_dependency_1Identity)gradients/huber_loss/Mul_3_grad/Reshape_11^gradients/huber_loss/Mul_3_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/huber_loss/Mul_3_grad/Reshape_1*
_output_shapes
: 
q
.gradients/huber_loss/Add_grad/tuple/group_depsNoOp9^gradients/huber_loss/Mul_3_grad/tuple/control_dependency
Т
6gradients/huber_loss/Add_grad/tuple/control_dependencyIdentity8gradients/huber_loss/Mul_3_grad/tuple/control_dependency/^gradients/huber_loss/Add_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/huber_loss/Mul_3_grad/Reshape*
_output_shapes

: 
Ф
8gradients/huber_loss/Add_grad/tuple/control_dependency_1Identity8gradients/huber_loss/Mul_3_grad/tuple/control_dependency/^gradients/huber_loss/Add_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/huber_loss/Mul_3_grad/Reshape*
_output_shapes

: 
h
%gradients/huber_loss/Mul_1_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
x
'gradients/huber_loss/Mul_1_grad/Shape_1Const*
valueB"       *
dtype0*
_output_shapes
:
џ
5gradients/huber_loss/Mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs%gradients/huber_loss/Mul_1_grad/Shape'gradients/huber_loss/Mul_1_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Ы
#gradients/huber_loss/Mul_1_grad/MulMul6gradients/huber_loss/Add_grad/tuple/control_dependencyhuber_loss/Mul*
T0*
_output_shapes

: 
∆
#gradients/huber_loss/Mul_1_grad/SumSum#gradients/huber_loss/Mul_1_grad/Mul5gradients/huber_loss/Mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
≠
'gradients/huber_loss/Mul_1_grad/ReshapeReshape#gradients/huber_loss/Mul_1_grad/Sum%gradients/huber_loss/Mul_1_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
Я
%gradients/huber_loss/Mul_1_grad/Mul_1Mulhuber_loss/Const6gradients/huber_loss/Add_grad/tuple/control_dependency*
T0*
_output_shapes

: 
ћ
%gradients/huber_loss/Mul_1_grad/Sum_1Sum%gradients/huber_loss/Mul_1_grad/Mul_17gradients/huber_loss/Mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ї
)gradients/huber_loss/Mul_1_grad/Reshape_1Reshape%gradients/huber_loss/Mul_1_grad/Sum_1'gradients/huber_loss/Mul_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes

: 
О
0gradients/huber_loss/Mul_1_grad/tuple/group_depsNoOp(^gradients/huber_loss/Mul_1_grad/Reshape*^gradients/huber_loss/Mul_1_grad/Reshape_1
э
8gradients/huber_loss/Mul_1_grad/tuple/control_dependencyIdentity'gradients/huber_loss/Mul_1_grad/Reshape1^gradients/huber_loss/Mul_1_grad/tuple/group_deps*
_output_shapes
: *
T0*:
_class0
.,loc:@gradients/huber_loss/Mul_1_grad/Reshape
Л
:gradients/huber_loss/Mul_1_grad/tuple/control_dependency_1Identity)gradients/huber_loss/Mul_1_grad/Reshape_11^gradients/huber_loss/Mul_1_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/huber_loss/Mul_1_grad/Reshape_1*
_output_shapes

: 
h
%gradients/huber_loss/Mul_2_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
x
'gradients/huber_loss/Mul_2_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB"       
џ
5gradients/huber_loss/Mul_2_grad/BroadcastGradientArgsBroadcastGradientArgs%gradients/huber_loss/Mul_2_grad/Shape'gradients/huber_loss/Mul_2_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
Я
#gradients/huber_loss/Mul_2_grad/MulMul8gradients/huber_loss/Add_grad/tuple/control_dependency_1huber_loss/Sub_1*
T0*
_output_shapes

: 
∆
#gradients/huber_loss/Mul_2_grad/SumSum#gradients/huber_loss/Mul_2_grad/Mul5gradients/huber_loss/Mul_2_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
≠
'gradients/huber_loss/Mul_2_grad/ReshapeReshape#gradients/huber_loss/Mul_2_grad/Sum%gradients/huber_loss/Mul_2_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
£
%gradients/huber_loss/Mul_2_grad/Mul_1Mulhuber_loss/Mul_2/x8gradients/huber_loss/Add_grad/tuple/control_dependency_1*
T0*
_output_shapes

: 
ћ
%gradients/huber_loss/Mul_2_grad/Sum_1Sum%gradients/huber_loss/Mul_2_grad/Mul_17gradients/huber_loss/Mul_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ї
)gradients/huber_loss/Mul_2_grad/Reshape_1Reshape%gradients/huber_loss/Mul_2_grad/Sum_1'gradients/huber_loss/Mul_2_grad/Shape_1*
_output_shapes

: *
T0*
Tshape0
О
0gradients/huber_loss/Mul_2_grad/tuple/group_depsNoOp(^gradients/huber_loss/Mul_2_grad/Reshape*^gradients/huber_loss/Mul_2_grad/Reshape_1
э
8gradients/huber_loss/Mul_2_grad/tuple/control_dependencyIdentity'gradients/huber_loss/Mul_2_grad/Reshape1^gradients/huber_loss/Mul_2_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/huber_loss/Mul_2_grad/Reshape*
_output_shapes
: 
Л
:gradients/huber_loss/Mul_2_grad/tuple/control_dependency_1Identity)gradients/huber_loss/Mul_2_grad/Reshape_11^gradients/huber_loss/Mul_2_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/huber_loss/Mul_2_grad/Reshape_1*
_output_shapes

: 
°
!gradients/huber_loss/Mul_grad/MulMul:gradients/huber_loss/Mul_1_grad/tuple/control_dependency_1huber_loss/Minimum*
T0*
_output_shapes

: 
£
#gradients/huber_loss/Mul_grad/Mul_1Mul:gradients/huber_loss/Mul_1_grad/tuple/control_dependency_1huber_loss/Minimum*
_output_shapes

: *
T0
А
.gradients/huber_loss/Mul_grad/tuple/group_depsNoOp"^gradients/huber_loss/Mul_grad/Mul$^gradients/huber_loss/Mul_grad/Mul_1
х
6gradients/huber_loss/Mul_grad/tuple/control_dependencyIdentity!gradients/huber_loss/Mul_grad/Mul/^gradients/huber_loss/Mul_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients/huber_loss/Mul_grad/Mul*
_output_shapes

: 
ы
8gradients/huber_loss/Mul_grad/tuple/control_dependency_1Identity#gradients/huber_loss/Mul_grad/Mul_1/^gradients/huber_loss/Mul_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients/huber_loss/Mul_grad/Mul_1*
_output_shapes

: 
П
#gradients/huber_loss/Sub_1_grad/NegNeg:gradients/huber_loss/Mul_2_grad/tuple/control_dependency_1*
_output_shapes

: *
T0
Ы
0gradients/huber_loss/Sub_1_grad/tuple/group_depsNoOp;^gradients/huber_loss/Mul_2_grad/tuple/control_dependency_1$^gradients/huber_loss/Sub_1_grad/Neg
Ъ
8gradients/huber_loss/Sub_1_grad/tuple/control_dependencyIdentity:gradients/huber_loss/Mul_2_grad/tuple/control_dependency_11^gradients/huber_loss/Sub_1_grad/tuple/group_deps*
_output_shapes

: *
T0*<
_class2
0.loc:@gradients/huber_loss/Mul_2_grad/Reshape_1
€
:gradients/huber_loss/Sub_1_grad/tuple/control_dependency_1Identity#gradients/huber_loss/Sub_1_grad/Neg1^gradients/huber_loss/Sub_1_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients/huber_loss/Sub_1_grad/Neg*
_output_shapes

: 
Ѓ
gradients/AddN_1AddN6gradients/huber_loss/Mul_grad/tuple/control_dependency8gradients/huber_loss/Mul_grad/tuple/control_dependency_1:gradients/huber_loss/Sub_1_grad/tuple/control_dependency_1*
T0*4
_class*
(&loc:@gradients/huber_loss/Mul_grad/Mul*
N*
_output_shapes

: 
x
'gradients/huber_loss/Minimum_grad/ShapeConst*
valueB"       *
dtype0*
_output_shapes
:
l
)gradients/huber_loss/Minimum_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
z
)gradients/huber_loss/Minimum_grad/Shape_2Const*
dtype0*
_output_shapes
:*
valueB"       
r
-gradients/huber_loss/Minimum_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ƒ
'gradients/huber_loss/Minimum_grad/zerosFill)gradients/huber_loss/Minimum_grad/Shape_2-gradients/huber_loss/Minimum_grad/zeros/Const*
_output_shapes

: *
T0*

index_type0
З
+gradients/huber_loss/Minimum_grad/LessEqual	LessEqualhuber_loss/Abshuber_loss/Minimum/y*
T0*
_output_shapes

: 
б
7gradients/huber_loss/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgs'gradients/huber_loss/Minimum_grad/Shape)gradients/huber_loss/Minimum_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
√
(gradients/huber_loss/Minimum_grad/SelectSelect+gradients/huber_loss/Minimum_grad/LessEqualgradients/AddN_1'gradients/huber_loss/Minimum_grad/zeros*
_output_shapes

: *
T0
≈
*gradients/huber_loss/Minimum_grad/Select_1Select+gradients/huber_loss/Minimum_grad/LessEqual'gradients/huber_loss/Minimum_grad/zerosgradients/AddN_1*
T0*
_output_shapes

: 
ѕ
%gradients/huber_loss/Minimum_grad/SumSum(gradients/huber_loss/Minimum_grad/Select7gradients/huber_loss/Minimum_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ї
)gradients/huber_loss/Minimum_grad/ReshapeReshape%gradients/huber_loss/Minimum_grad/Sum'gradients/huber_loss/Minimum_grad/Shape*
_output_shapes

: *
T0*
Tshape0
’
'gradients/huber_loss/Minimum_grad/Sum_1Sum*gradients/huber_loss/Minimum_grad/Select_19gradients/huber_loss/Minimum_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
є
+gradients/huber_loss/Minimum_grad/Reshape_1Reshape'gradients/huber_loss/Minimum_grad/Sum_1)gradients/huber_loss/Minimum_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
Ф
2gradients/huber_loss/Minimum_grad/tuple/group_depsNoOp*^gradients/huber_loss/Minimum_grad/Reshape,^gradients/huber_loss/Minimum_grad/Reshape_1
Н
:gradients/huber_loss/Minimum_grad/tuple/control_dependencyIdentity)gradients/huber_loss/Minimum_grad/Reshape3^gradients/huber_loss/Minimum_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/huber_loss/Minimum_grad/Reshape*
_output_shapes

: 
Л
<gradients/huber_loss/Minimum_grad/tuple/control_dependency_1Identity+gradients/huber_loss/Minimum_grad/Reshape_13^gradients/huber_loss/Minimum_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/huber_loss/Minimum_grad/Reshape_1*
_output_shapes
: 
ю
gradients/AddN_2AddN8gradients/huber_loss/Sub_1_grad/tuple/control_dependency:gradients/huber_loss/Minimum_grad/tuple/control_dependency*
T0*<
_class2
0.loc:@gradients/huber_loss/Mul_2_grad/Reshape_1*
N*
_output_shapes

: 
c
"gradients/huber_loss/Abs_grad/SignSignhuber_loss/Sub*
T0*
_output_shapes

: 
З
!gradients/huber_loss/Abs_grad/mulMulgradients/AddN_2"gradients/huber_loss/Abs_grad/Sign*
T0*
_output_shapes

: 
f
#gradients/huber_loss/Sub_grad/ShapeShapeSum*
T0*
out_type0*
_output_shapes
:
r
%gradients/huber_loss/Sub_grad/Shape_1ShapePlaceholder_2*
T0*
out_type0*
_output_shapes
:
’
3gradients/huber_loss/Sub_grad/BroadcastGradientArgsBroadcastGradientArgs#gradients/huber_loss/Sub_grad/Shape%gradients/huber_loss/Sub_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
ј
!gradients/huber_loss/Sub_grad/SumSum!gradients/huber_loss/Abs_grad/mul3gradients/huber_loss/Sub_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Є
%gradients/huber_loss/Sub_grad/ReshapeReshape!gradients/huber_loss/Sub_grad/Sum#gradients/huber_loss/Sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
: €€€€€€€€€
ƒ
#gradients/huber_loss/Sub_grad/Sum_1Sum!gradients/huber_loss/Abs_grad/mul5gradients/huber_loss/Sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
p
!gradients/huber_loss/Sub_grad/NegNeg#gradients/huber_loss/Sub_grad/Sum_1*
_output_shapes
:*
T0
Љ
'gradients/huber_loss/Sub_grad/Reshape_1Reshape!gradients/huber_loss/Sub_grad/Neg%gradients/huber_loss/Sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
И
.gradients/huber_loss/Sub_grad/tuple/group_depsNoOp&^gradients/huber_loss/Sub_grad/Reshape(^gradients/huber_loss/Sub_grad/Reshape_1
Ж
6gradients/huber_loss/Sub_grad/tuple/control_dependencyIdentity%gradients/huber_loss/Sub_grad/Reshape/^gradients/huber_loss/Sub_grad/tuple/group_deps*
T0*8
_class.
,*loc:@gradients/huber_loss/Sub_grad/Reshape*'
_output_shapes
: €€€€€€€€€
М
8gradients/huber_loss/Sub_grad/tuple/control_dependency_1Identity'gradients/huber_loss/Sub_grad/Reshape_1/^gradients/huber_loss/Sub_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/huber_loss/Sub_grad/Reshape_1*'
_output_shapes
:€€€€€€€€€
[
gradients/Sum_grad/ShapeShapemul*
T0*
out_type0*
_output_shapes
:
Ж
gradients/Sum_grad/SizeConst*+
_class!
loc:@gradients/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
Ы
gradients/Sum_grad/addAddSum/reduction_indicesgradients/Sum_grad/Size*
_output_shapes
: *
T0*+
_class!
loc:@gradients/Sum_grad/Shape
°
gradients/Sum_grad/modFloorModgradients/Sum_grad/addgradients/Sum_grad/Size*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
: 
К
gradients/Sum_grad/Shape_1Const*+
_class!
loc:@gradients/Sum_grad/Shape*
valueB *
dtype0*
_output_shapes
: 
Н
gradients/Sum_grad/range/startConst*+
_class!
loc:@gradients/Sum_grad/Shape*
value	B : *
dtype0*
_output_shapes
: 
Н
gradients/Sum_grad/range/deltaConst*+
_class!
loc:@gradients/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
ѕ
gradients/Sum_grad/rangeRangegradients/Sum_grad/range/startgradients/Sum_grad/Sizegradients/Sum_grad/range/delta*

Tidx0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
:
М
gradients/Sum_grad/Fill/valueConst*+
_class!
loc:@gradients/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
Ї
gradients/Sum_grad/FillFillgradients/Sum_grad/Shape_1gradients/Sum_grad/Fill/value*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*

index_type0*
_output_shapes
: 
ъ
 gradients/Sum_grad/DynamicStitchDynamicStitchgradients/Sum_grad/rangegradients/Sum_grad/modgradients/Sum_grad/Shapegradients/Sum_grad/Fill*
N*#
_output_shapes
:€€€€€€€€€*
T0*+
_class!
loc:@gradients/Sum_grad/Shape
Л
gradients/Sum_grad/Maximum/yConst*+
_class!
loc:@gradients/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
ј
gradients/Sum_grad/MaximumMaximum gradients/Sum_grad/DynamicStitchgradients/Sum_grad/Maximum/y*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*#
_output_shapes
:€€€€€€€€€
ѓ
gradients/Sum_grad/floordivFloorDivgradients/Sum_grad/Shapegradients/Sum_grad/Maximum*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
:
∞
gradients/Sum_grad/ReshapeReshape6gradients/huber_loss/Sub_grad/tuple/control_dependency gradients/Sum_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
†
gradients/Sum_grad/TileTilegradients/Sum_grad/Reshapegradients/Sum_grad/floordiv*

Tmultiples0*
T0*+
_output_shapes
: €€€€€€€€€
f
gradients/mul_grad/ShapeShapemain/transpose*
T0*
out_type0*
_output_shapes
:
d
gradients/mul_grad/Shape_1Shape
ExpandDims*
_output_shapes
:*
T0*
out_type0
і
(gradients/mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_grad/Shapegradients/mul_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
x
gradients/mul_grad/MulMulgradients/Sum_grad/Tile
ExpandDims*+
_output_shapes
: €€€€€€€€€*
T0
Я
gradients/mul_grad/SumSumgradients/mul_grad/Mul(gradients/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ы
gradients/mul_grad/ReshapeReshapegradients/mul_grad/Sumgradients/mul_grad/Shape*
T0*
Tshape0*+
_output_shapes
: €€€€€€€€€
~
gradients/mul_grad/Mul_1Mulmain/transposegradients/Sum_grad/Tile*
T0*+
_output_shapes
: €€€€€€€€€
•
gradients/mul_grad/Sum_1Sumgradients/mul_grad/Mul_1*gradients/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
°
gradients/mul_grad/Reshape_1Reshapegradients/mul_grad/Sum_1gradients/mul_grad/Shape_1*
T0*
Tshape0*+
_output_shapes
:€€€€€€€€€
g
#gradients/mul_grad/tuple/group_depsNoOp^gradients/mul_grad/Reshape^gradients/mul_grad/Reshape_1
ё
+gradients/mul_grad/tuple/control_dependencyIdentitygradients/mul_grad/Reshape$^gradients/mul_grad/tuple/group_deps*+
_output_shapes
: €€€€€€€€€*
T0*-
_class#
!loc:@gradients/mul_grad/Reshape
д
-gradients/mul_grad/tuple/control_dependency_1Identitygradients/mul_grad/Reshape_1$^gradients/mul_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/mul_grad/Reshape_1*+
_output_shapes
:€€€€€€€€€
~
/gradients/main/transpose_grad/InvertPermutationInvertPermutationmain/transpose/perm*
_output_shapes
:*
T0
’
'gradients/main/transpose_grad/transpose	Transpose+gradients/mul_grad/tuple/control_dependency/gradients/main/transpose_grad/InvertPermutation*+
_output_shapes
: €€€€€€€€€*
Tperm0*
T0
у
'gradients/main/transpose/x_grad/unstackUnpack'gradients/main/transpose_grad/transpose*	
num *
T0*

axis *ц
_output_shapesг
а:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€
b
0gradients/main/transpose/x_grad/tuple/group_depsNoOp(^gradients/main/transpose/x_grad/unstack
О
8gradients/main/transpose/x_grad/tuple/control_dependencyIdentity'gradients/main/transpose/x_grad/unstack1^gradients/main/transpose/x_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack
Т
:gradients/main/transpose/x_grad/tuple/control_dependency_1Identity)gradients/main/transpose/x_grad/unstack:11^gradients/main/transpose/x_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack
Т
:gradients/main/transpose/x_grad/tuple/control_dependency_2Identity)gradients/main/transpose/x_grad/unstack:21^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:€€€€€€€€€
Т
:gradients/main/transpose/x_grad/tuple/control_dependency_3Identity)gradients/main/transpose/x_grad/unstack:31^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:€€€€€€€€€
Т
:gradients/main/transpose/x_grad/tuple/control_dependency_4Identity)gradients/main/transpose/x_grad/unstack:41^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:€€€€€€€€€
Т
:gradients/main/transpose/x_grad/tuple/control_dependency_5Identity)gradients/main/transpose/x_grad/unstack:51^gradients/main/transpose/x_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack
Т
:gradients/main/transpose/x_grad/tuple/control_dependency_6Identity)gradients/main/transpose/x_grad/unstack:61^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:€€€€€€€€€
Т
:gradients/main/transpose/x_grad/tuple/control_dependency_7Identity)gradients/main/transpose/x_grad/unstack:71^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:€€€€€€€€€
Т
:gradients/main/transpose/x_grad/tuple/control_dependency_8Identity)gradients/main/transpose/x_grad/unstack:81^gradients/main/transpose/x_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack
Т
:gradients/main/transpose/x_grad/tuple/control_dependency_9Identity)gradients/main/transpose/x_grad/unstack:91^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:€€€€€€€€€
Ф
;gradients/main/transpose/x_grad/tuple/control_dependency_10Identity*gradients/main/transpose/x_grad/unstack:101^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:€€€€€€€€€
Ф
;gradients/main/transpose/x_grad/tuple/control_dependency_11Identity*gradients/main/transpose/x_grad/unstack:111^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:€€€€€€€€€
Ф
;gradients/main/transpose/x_grad/tuple/control_dependency_12Identity*gradients/main/transpose/x_grad/unstack:121^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:€€€€€€€€€
Ф
;gradients/main/transpose/x_grad/tuple/control_dependency_13Identity*gradients/main/transpose/x_grad/unstack:131^gradients/main/transpose/x_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack
Ф
;gradients/main/transpose/x_grad/tuple/control_dependency_14Identity*gradients/main/transpose/x_grad/unstack:141^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:€€€€€€€€€
Ф
;gradients/main/transpose/x_grad/tuple/control_dependency_15Identity*gradients/main/transpose/x_grad/unstack:151^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:€€€€€€€€€
Ф
;gradients/main/transpose/x_grad/tuple/control_dependency_16Identity*gradients/main/transpose/x_grad/unstack:161^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:€€€€€€€€€
Ф
;gradients/main/transpose/x_grad/tuple/control_dependency_17Identity*gradients/main/transpose/x_grad/unstack:171^gradients/main/transpose/x_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack
Ф
;gradients/main/transpose/x_grad/tuple/control_dependency_18Identity*gradients/main/transpose/x_grad/unstack:181^gradients/main/transpose/x_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack
Ф
;gradients/main/transpose/x_grad/tuple/control_dependency_19Identity*gradients/main/transpose/x_grad/unstack:191^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:€€€€€€€€€
Ф
;gradients/main/transpose/x_grad/tuple/control_dependency_20Identity*gradients/main/transpose/x_grad/unstack:201^gradients/main/transpose/x_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack
Ф
;gradients/main/transpose/x_grad/tuple/control_dependency_21Identity*gradients/main/transpose/x_grad/unstack:211^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:€€€€€€€€€
Ф
;gradients/main/transpose/x_grad/tuple/control_dependency_22Identity*gradients/main/transpose/x_grad/unstack:221^gradients/main/transpose/x_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack
Ф
;gradients/main/transpose/x_grad/tuple/control_dependency_23Identity*gradients/main/transpose/x_grad/unstack:231^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:€€€€€€€€€
Ф
;gradients/main/transpose/x_grad/tuple/control_dependency_24Identity*gradients/main/transpose/x_grad/unstack:241^gradients/main/transpose/x_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack
Ф
;gradients/main/transpose/x_grad/tuple/control_dependency_25Identity*gradients/main/transpose/x_grad/unstack:251^gradients/main/transpose/x_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack
Ф
;gradients/main/transpose/x_grad/tuple/control_dependency_26Identity*gradients/main/transpose/x_grad/unstack:261^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:€€€€€€€€€
Ф
;gradients/main/transpose/x_grad/tuple/control_dependency_27Identity*gradients/main/transpose/x_grad/unstack:271^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:€€€€€€€€€
Ф
;gradients/main/transpose/x_grad/tuple/control_dependency_28Identity*gradients/main/transpose/x_grad/unstack:281^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:€€€€€€€€€
Ф
;gradients/main/transpose/x_grad/tuple/control_dependency_29Identity*gradients/main/transpose/x_grad/unstack:291^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:€€€€€€€€€
Ф
;gradients/main/transpose/x_grad/tuple/control_dependency_30Identity*gradients/main/transpose/x_grad/unstack:301^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:€€€€€€€€€
Ф
;gradients/main/transpose/x_grad/tuple/control_dependency_31Identity*gradients/main/transpose/x_grad/unstack:311^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:€€€€€€€€€
Э
 gradients/main/split_grad/concatConcatV28gradients/main/transpose/x_grad/tuple/control_dependency:gradients/main/transpose/x_grad/tuple/control_dependency_1:gradients/main/transpose/x_grad/tuple/control_dependency_2:gradients/main/transpose/x_grad/tuple/control_dependency_3:gradients/main/transpose/x_grad/tuple/control_dependency_4:gradients/main/transpose/x_grad/tuple/control_dependency_5:gradients/main/transpose/x_grad/tuple/control_dependency_6:gradients/main/transpose/x_grad/tuple/control_dependency_7:gradients/main/transpose/x_grad/tuple/control_dependency_8:gradients/main/transpose/x_grad/tuple/control_dependency_9;gradients/main/transpose/x_grad/tuple/control_dependency_10;gradients/main/transpose/x_grad/tuple/control_dependency_11;gradients/main/transpose/x_grad/tuple/control_dependency_12;gradients/main/transpose/x_grad/tuple/control_dependency_13;gradients/main/transpose/x_grad/tuple/control_dependency_14;gradients/main/transpose/x_grad/tuple/control_dependency_15;gradients/main/transpose/x_grad/tuple/control_dependency_16;gradients/main/transpose/x_grad/tuple/control_dependency_17;gradients/main/transpose/x_grad/tuple/control_dependency_18;gradients/main/transpose/x_grad/tuple/control_dependency_19;gradients/main/transpose/x_grad/tuple/control_dependency_20;gradients/main/transpose/x_grad/tuple/control_dependency_21;gradients/main/transpose/x_grad/tuple/control_dependency_22;gradients/main/transpose/x_grad/tuple/control_dependency_23;gradients/main/transpose/x_grad/tuple/control_dependency_24;gradients/main/transpose/x_grad/tuple/control_dependency_25;gradients/main/transpose/x_grad/tuple/control_dependency_26;gradients/main/transpose/x_grad/tuple/control_dependency_27;gradients/main/transpose/x_grad/tuple/control_dependency_28;gradients/main/transpose/x_grad/tuple/control_dependency_29;gradients/main/transpose/x_grad/tuple/control_dependency_30;gradients/main/transpose/x_grad/tuple/control_dependency_31main/split/split_dim*
N *'
_output_shapes
:€€€€€€€€€*

Tidx0*
T0
Ь
/gradients/main/dense_4/BiasAdd_grad/BiasAddGradBiasAddGrad gradients/main/split_grad/concat*
T0*
data_formatNHWC*
_output_shapes
:
С
4gradients/main/dense_4/BiasAdd_grad/tuple/group_depsNoOp0^gradients/main/dense_4/BiasAdd_grad/BiasAddGrad!^gradients/main/split_grad/concat
И
<gradients/main/dense_4/BiasAdd_grad/tuple/control_dependencyIdentity gradients/main/split_grad/concat5^gradients/main/dense_4/BiasAdd_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/main/split_grad/concat*'
_output_shapes
:€€€€€€€€€
Ы
>gradients/main/dense_4/BiasAdd_grad/tuple/control_dependency_1Identity/gradients/main/dense_4/BiasAdd_grad/BiasAddGrad5^gradients/main/dense_4/BiasAdd_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/main/dense_4/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
д
)gradients/main/dense_4/MatMul_grad/MatMulMatMul<gradients/main/dense_4/BiasAdd_grad/tuple/control_dependencymain/dense_4/kernel/read*(
_output_shapes
:€€€€€€€€€А*
transpose_a( *
transpose_b(*
T0
÷
+gradients/main/dense_4/MatMul_grad/MatMul_1MatMulmain/dense_3/Relu<gradients/main/dense_4/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	А*
transpose_a(*
transpose_b( 
Х
3gradients/main/dense_4/MatMul_grad/tuple/group_depsNoOp*^gradients/main/dense_4/MatMul_grad/MatMul,^gradients/main/dense_4/MatMul_grad/MatMul_1
Щ
;gradients/main/dense_4/MatMul_grad/tuple/control_dependencyIdentity)gradients/main/dense_4/MatMul_grad/MatMul4^gradients/main/dense_4/MatMul_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/main/dense_4/MatMul_grad/MatMul*(
_output_shapes
:€€€€€€€€€А
Ц
=gradients/main/dense_4/MatMul_grad/tuple/control_dependency_1Identity+gradients/main/dense_4/MatMul_grad/MatMul_14^gradients/main/dense_4/MatMul_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/main/dense_4/MatMul_grad/MatMul_1*
_output_shapes
:	А
Є
)gradients/main/dense_3/Relu_grad/ReluGradReluGrad;gradients/main/dense_4/MatMul_grad/tuple/control_dependencymain/dense_3/Relu*
T0*(
_output_shapes
:€€€€€€€€€А
¶
/gradients/main/dense_3/BiasAdd_grad/BiasAddGradBiasAddGrad)gradients/main/dense_3/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:А
Ъ
4gradients/main/dense_3/BiasAdd_grad/tuple/group_depsNoOp0^gradients/main/dense_3/BiasAdd_grad/BiasAddGrad*^gradients/main/dense_3/Relu_grad/ReluGrad
Ы
<gradients/main/dense_3/BiasAdd_grad/tuple/control_dependencyIdentity)gradients/main/dense_3/Relu_grad/ReluGrad5^gradients/main/dense_3/BiasAdd_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/main/dense_3/Relu_grad/ReluGrad*(
_output_shapes
:€€€€€€€€€А
Ь
>gradients/main/dense_3/BiasAdd_grad/tuple/control_dependency_1Identity/gradients/main/dense_3/BiasAdd_grad/BiasAddGrad5^gradients/main/dense_3/BiasAdd_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/main/dense_3/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:А
д
)gradients/main/dense_3/MatMul_grad/MatMulMatMul<gradients/main/dense_3/BiasAdd_grad/tuple/control_dependencymain/dense_3/kernel/read*
T0*(
_output_shapes
:€€€€€€€€€А*
transpose_a( *
transpose_b(
„
+gradients/main/dense_3/MatMul_grad/MatMul_1MatMulmain/dense_2/Relu<gradients/main/dense_3/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
АА*
transpose_a(*
transpose_b( *
T0
Х
3gradients/main/dense_3/MatMul_grad/tuple/group_depsNoOp*^gradients/main/dense_3/MatMul_grad/MatMul,^gradients/main/dense_3/MatMul_grad/MatMul_1
Щ
;gradients/main/dense_3/MatMul_grad/tuple/control_dependencyIdentity)gradients/main/dense_3/MatMul_grad/MatMul4^gradients/main/dense_3/MatMul_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/main/dense_3/MatMul_grad/MatMul*(
_output_shapes
:€€€€€€€€€А
Ч
=gradients/main/dense_3/MatMul_grad/tuple/control_dependency_1Identity+gradients/main/dense_3/MatMul_grad/MatMul_14^gradients/main/dense_3/MatMul_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/main/dense_3/MatMul_grad/MatMul_1* 
_output_shapes
:
АА
Є
)gradients/main/dense_2/Relu_grad/ReluGradReluGrad;gradients/main/dense_3/MatMul_grad/tuple/control_dependencymain/dense_2/Relu*(
_output_shapes
:€€€€€€€€€А*
T0
¶
/gradients/main/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad)gradients/main/dense_2/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:А
Ъ
4gradients/main/dense_2/BiasAdd_grad/tuple/group_depsNoOp0^gradients/main/dense_2/BiasAdd_grad/BiasAddGrad*^gradients/main/dense_2/Relu_grad/ReluGrad
Ы
<gradients/main/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity)gradients/main/dense_2/Relu_grad/ReluGrad5^gradients/main/dense_2/BiasAdd_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/main/dense_2/Relu_grad/ReluGrad*(
_output_shapes
:€€€€€€€€€А
Ь
>gradients/main/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity/gradients/main/dense_2/BiasAdd_grad/BiasAddGrad5^gradients/main/dense_2/BiasAdd_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/main/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:А
д
)gradients/main/dense_2/MatMul_grad/MatMulMatMul<gradients/main/dense_2/BiasAdd_grad/tuple/control_dependencymain/dense_2/kernel/read*
T0*(
_output_shapes
:€€€€€€€€€А*
transpose_a( *
transpose_b(
ќ
+gradients/main/dense_2/MatMul_grad/MatMul_1MatMulmain/Mul<gradients/main/dense_2/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
АА*
transpose_a(*
transpose_b( *
T0
Х
3gradients/main/dense_2/MatMul_grad/tuple/group_depsNoOp*^gradients/main/dense_2/MatMul_grad/MatMul,^gradients/main/dense_2/MatMul_grad/MatMul_1
Щ
;gradients/main/dense_2/MatMul_grad/tuple/control_dependencyIdentity)gradients/main/dense_2/MatMul_grad/MatMul4^gradients/main/dense_2/MatMul_grad/tuple/group_deps*(
_output_shapes
:€€€€€€€€€А*
T0*<
_class2
0.loc:@gradients/main/dense_2/MatMul_grad/MatMul
Ч
=gradients/main/dense_2/MatMul_grad/tuple/control_dependency_1Identity+gradients/main/dense_2/MatMul_grad/MatMul_14^gradients/main/dense_2/MatMul_grad/tuple/group_deps* 
_output_shapes
:
АА*
T0*>
_class4
20loc:@gradients/main/dense_2/MatMul_grad/MatMul_1
l
gradients/main/Mul_grad/ShapeShapemain/dense/Selu*
_output_shapes
:*
T0*
out_type0
p
gradients/main/Mul_grad/Shape_1Shapemain/dense_1/Relu*
_output_shapes
:*
T0*
out_type0
√
-gradients/main/Mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/main/Mul_grad/Shapegradients/main/Mul_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
•
gradients/main/Mul_grad/MulMul;gradients/main/dense_2/MatMul_grad/tuple/control_dependencymain/dense_1/Relu*
T0*(
_output_shapes
:€€€€€€€€€А
Ѓ
gradients/main/Mul_grad/SumSumgradients/main/Mul_grad/Mul-gradients/main/Mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
І
gradients/main/Mul_grad/ReshapeReshapegradients/main/Mul_grad/Sumgradients/main/Mul_grad/Shape*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
•
gradients/main/Mul_grad/Mul_1Mulmain/dense/Selu;gradients/main/dense_2/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:€€€€€€€€€А
і
gradients/main/Mul_grad/Sum_1Sumgradients/main/Mul_grad/Mul_1/gradients/main/Mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
≠
!gradients/main/Mul_grad/Reshape_1Reshapegradients/main/Mul_grad/Sum_1gradients/main/Mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
v
(gradients/main/Mul_grad/tuple/group_depsNoOp ^gradients/main/Mul_grad/Reshape"^gradients/main/Mul_grad/Reshape_1
п
0gradients/main/Mul_grad/tuple/control_dependencyIdentitygradients/main/Mul_grad/Reshape)^gradients/main/Mul_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/main/Mul_grad/Reshape*(
_output_shapes
:€€€€€€€€€А
х
2gradients/main/Mul_grad/tuple/control_dependency_1Identity!gradients/main/Mul_grad/Reshape_1)^gradients/main/Mul_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients/main/Mul_grad/Reshape_1*(
_output_shapes
:€€€€€€€€€А
©
'gradients/main/dense/Selu_grad/SeluGradSeluGrad0gradients/main/Mul_grad/tuple/control_dependencymain/dense/Selu*(
_output_shapes
:€€€€€€€€€А*
T0
ѓ
)gradients/main/dense_1/Relu_grad/ReluGradReluGrad2gradients/main/Mul_grad/tuple/control_dependency_1main/dense_1/Relu*
T0*(
_output_shapes
:€€€€€€€€€А
Ґ
-gradients/main/dense/BiasAdd_grad/BiasAddGradBiasAddGrad'gradients/main/dense/Selu_grad/SeluGrad*
T0*
data_formatNHWC*
_output_shapes	
:А
Ф
2gradients/main/dense/BiasAdd_grad/tuple/group_depsNoOp.^gradients/main/dense/BiasAdd_grad/BiasAddGrad(^gradients/main/dense/Selu_grad/SeluGrad
У
:gradients/main/dense/BiasAdd_grad/tuple/control_dependencyIdentity'gradients/main/dense/Selu_grad/SeluGrad3^gradients/main/dense/BiasAdd_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/dense/Selu_grad/SeluGrad*(
_output_shapes
:€€€€€€€€€А
Ф
<gradients/main/dense/BiasAdd_grad/tuple/control_dependency_1Identity-gradients/main/dense/BiasAdd_grad/BiasAddGrad3^gradients/main/dense/BiasAdd_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/main/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:А
¶
/gradients/main/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad)gradients/main/dense_1/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes	
:А*
T0
Ъ
4gradients/main/dense_1/BiasAdd_grad/tuple/group_depsNoOp0^gradients/main/dense_1/BiasAdd_grad/BiasAddGrad*^gradients/main/dense_1/Relu_grad/ReluGrad
Ы
<gradients/main/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity)gradients/main/dense_1/Relu_grad/ReluGrad5^gradients/main/dense_1/BiasAdd_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/main/dense_1/Relu_grad/ReluGrad*(
_output_shapes
:€€€€€€€€€А
Ь
>gradients/main/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity/gradients/main/dense_1/BiasAdd_grad/BiasAddGrad5^gradients/main/dense_1/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:А*
T0*B
_class8
64loc:@gradients/main/dense_1/BiasAdd_grad/BiasAddGrad
Ё
'gradients/main/dense/MatMul_grad/MatMulMatMul:gradients/main/dense/BiasAdd_grad/tuple/control_dependencymain/dense/kernel/read*
T0*'
_output_shapes
:€€€€€€€€€*
transpose_a( *
transpose_b(
Ќ
)gradients/main/dense/MatMul_grad/MatMul_1MatMulmain/Reshape:gradients/main/dense/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	А*
transpose_a(*
transpose_b( 
П
1gradients/main/dense/MatMul_grad/tuple/group_depsNoOp(^gradients/main/dense/MatMul_grad/MatMul*^gradients/main/dense/MatMul_grad/MatMul_1
Р
9gradients/main/dense/MatMul_grad/tuple/control_dependencyIdentity'gradients/main/dense/MatMul_grad/MatMul2^gradients/main/dense/MatMul_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/dense/MatMul_grad/MatMul*'
_output_shapes
:€€€€€€€€€
О
;gradients/main/dense/MatMul_grad/tuple/control_dependency_1Identity)gradients/main/dense/MatMul_grad/MatMul_12^gradients/main/dense/MatMul_grad/tuple/group_deps*
_output_shapes
:	А*
T0*<
_class2
0.loc:@gradients/main/dense/MatMul_grad/MatMul_1
г
)gradients/main/dense_1/MatMul_grad/MatMulMatMul<gradients/main/dense_1/BiasAdd_grad/tuple/control_dependencymain/dense_1/kernel/read*
T0*'
_output_shapes
:€€€€€€€€€@*
transpose_a( *
transpose_b(
Ќ
+gradients/main/dense_1/MatMul_grad/MatMul_1MatMulmain/Cos<gradients/main/dense_1/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes
:	@А*
transpose_a(
Х
3gradients/main/dense_1/MatMul_grad/tuple/group_depsNoOp*^gradients/main/dense_1/MatMul_grad/MatMul,^gradients/main/dense_1/MatMul_grad/MatMul_1
Ш
;gradients/main/dense_1/MatMul_grad/tuple/control_dependencyIdentity)gradients/main/dense_1/MatMul_grad/MatMul4^gradients/main/dense_1/MatMul_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/main/dense_1/MatMul_grad/MatMul*'
_output_shapes
:€€€€€€€€€@
Ц
=gradients/main/dense_1/MatMul_grad/tuple/control_dependency_1Identity+gradients/main/dense_1/MatMul_grad/MatMul_14^gradients/main/dense_1/MatMul_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/main/dense_1/MatMul_grad/MatMul_1*
_output_shapes
:	@А
В
beta1_power/initial_valueConst*
dtype0*
_output_shapes
: *"
_class
loc:@main/dense/bias*
valueB
 *fff?
У
beta1_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name *"
_class
loc:@main/dense/bias*
	container *
shape: 
≤
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
use_locking(*
T0*"
_class
loc:@main/dense/bias*
validate_shape(*
_output_shapes
: 
n
beta1_power/readIdentitybeta1_power*
_output_shapes
: *
T0*"
_class
loc:@main/dense/bias
В
beta2_power/initial_valueConst*"
_class
loc:@main/dense/bias*
valueB
 *wЊ?*
dtype0*
_output_shapes
: 
У
beta2_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name *"
_class
loc:@main/dense/bias*
	container *
shape: 
≤
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*"
_class
loc:@main/dense/bias
n
beta2_power/readIdentitybeta2_power*
_output_shapes
: *
T0*"
_class
loc:@main/dense/bias
•
(main/dense/kernel/Adam/Initializer/zerosConst*$
_class
loc:@main/dense/kernel*
valueB	А*    *
dtype0*
_output_shapes
:	А
≤
main/dense/kernel/Adam
VariableV2*
shared_name *$
_class
loc:@main/dense/kernel*
	container *
shape:	А*
dtype0*
_output_shapes
:	А
в
main/dense/kernel/Adam/AssignAssignmain/dense/kernel/Adam(main/dense/kernel/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:	А*
use_locking(*
T0*$
_class
loc:@main/dense/kernel
П
main/dense/kernel/Adam/readIdentitymain/dense/kernel/Adam*
T0*$
_class
loc:@main/dense/kernel*
_output_shapes
:	А
І
*main/dense/kernel/Adam_1/Initializer/zerosConst*$
_class
loc:@main/dense/kernel*
valueB	А*    *
dtype0*
_output_shapes
:	А
і
main/dense/kernel/Adam_1
VariableV2*
dtype0*
_output_shapes
:	А*
shared_name *$
_class
loc:@main/dense/kernel*
	container *
shape:	А
и
main/dense/kernel/Adam_1/AssignAssignmain/dense/kernel/Adam_1*main/dense/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@main/dense/kernel*
validate_shape(*
_output_shapes
:	А
У
main/dense/kernel/Adam_1/readIdentitymain/dense/kernel/Adam_1*
T0*$
_class
loc:@main/dense/kernel*
_output_shapes
:	А
Щ
&main/dense/bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes	
:А*"
_class
loc:@main/dense/bias*
valueBА*    
¶
main/dense/bias/Adam
VariableV2*
shape:А*
dtype0*
_output_shapes	
:А*
shared_name *"
_class
loc:@main/dense/bias*
	container 
÷
main/dense/bias/Adam/AssignAssignmain/dense/bias/Adam&main/dense/bias/Adam/Initializer/zeros*
T0*"
_class
loc:@main/dense/bias*
validate_shape(*
_output_shapes	
:А*
use_locking(
Е
main/dense/bias/Adam/readIdentitymain/dense/bias/Adam*
T0*"
_class
loc:@main/dense/bias*
_output_shapes	
:А
Ы
(main/dense/bias/Adam_1/Initializer/zerosConst*"
_class
loc:@main/dense/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
®
main/dense/bias/Adam_1
VariableV2*
shape:А*
dtype0*
_output_shapes	
:А*
shared_name *"
_class
loc:@main/dense/bias*
	container 
№
main/dense/bias/Adam_1/AssignAssignmain/dense/bias/Adam_1(main/dense/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@main/dense/bias*
validate_shape(*
_output_shapes	
:А
Й
main/dense/bias/Adam_1/readIdentitymain/dense/bias/Adam_1*
_output_shapes	
:А*
T0*"
_class
loc:@main/dense/bias
≥
:main/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*&
_class
loc:@main/dense_1/kernel*
valueB"@   А   *
dtype0*
_output_shapes
:
Э
0main/dense_1/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *&
_class
loc:@main/dense_1/kernel*
valueB
 *    
Д
*main/dense_1/kernel/Adam/Initializer/zerosFill:main/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensor0main/dense_1/kernel/Adam/Initializer/zeros/Const*
_output_shapes
:	@А*
T0*&
_class
loc:@main/dense_1/kernel*

index_type0
ґ
main/dense_1/kernel/Adam
VariableV2*
shape:	@А*
dtype0*
_output_shapes
:	@А*
shared_name *&
_class
loc:@main/dense_1/kernel*
	container 
к
main/dense_1/kernel/Adam/AssignAssignmain/dense_1/kernel/Adam*main/dense_1/kernel/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:	@А*
use_locking(*
T0*&
_class
loc:@main/dense_1/kernel
Х
main/dense_1/kernel/Adam/readIdentitymain/dense_1/kernel/Adam*
_output_shapes
:	@А*
T0*&
_class
loc:@main/dense_1/kernel
µ
<main/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*&
_class
loc:@main/dense_1/kernel*
valueB"@   А   *
dtype0*
_output_shapes
:
Я
2main/dense_1/kernel/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *&
_class
loc:@main/dense_1/kernel*
valueB
 *    
К
,main/dense_1/kernel/Adam_1/Initializer/zerosFill<main/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor2main/dense_1/kernel/Adam_1/Initializer/zeros/Const*
T0*&
_class
loc:@main/dense_1/kernel*

index_type0*
_output_shapes
:	@А
Є
main/dense_1/kernel/Adam_1
VariableV2*
shared_name *&
_class
loc:@main/dense_1/kernel*
	container *
shape:	@А*
dtype0*
_output_shapes
:	@А
р
!main/dense_1/kernel/Adam_1/AssignAssignmain/dense_1/kernel/Adam_1,main/dense_1/kernel/Adam_1/Initializer/zeros*
T0*&
_class
loc:@main/dense_1/kernel*
validate_shape(*
_output_shapes
:	@А*
use_locking(
Щ
main/dense_1/kernel/Adam_1/readIdentitymain/dense_1/kernel/Adam_1*
T0*&
_class
loc:@main/dense_1/kernel*
_output_shapes
:	@А
Э
(main/dense_1/bias/Adam/Initializer/zerosConst*$
_class
loc:@main/dense_1/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
™
main/dense_1/bias/Adam
VariableV2*
shape:А*
dtype0*
_output_shapes	
:А*
shared_name *$
_class
loc:@main/dense_1/bias*
	container 
ё
main/dense_1/bias/Adam/AssignAssignmain/dense_1/bias/Adam(main/dense_1/bias/Adam/Initializer/zeros*
T0*$
_class
loc:@main/dense_1/bias*
validate_shape(*
_output_shapes	
:А*
use_locking(
Л
main/dense_1/bias/Adam/readIdentitymain/dense_1/bias/Adam*
T0*$
_class
loc:@main/dense_1/bias*
_output_shapes	
:А
Я
*main/dense_1/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes	
:А*$
_class
loc:@main/dense_1/bias*
valueBА*    
ђ
main/dense_1/bias/Adam_1
VariableV2*
dtype0*
_output_shapes	
:А*
shared_name *$
_class
loc:@main/dense_1/bias*
	container *
shape:А
д
main/dense_1/bias/Adam_1/AssignAssignmain/dense_1/bias/Adam_1*main/dense_1/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@main/dense_1/bias*
validate_shape(*
_output_shapes	
:А
П
main/dense_1/bias/Adam_1/readIdentitymain/dense_1/bias/Adam_1*
_output_shapes	
:А*
T0*$
_class
loc:@main/dense_1/bias
≥
:main/dense_2/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*&
_class
loc:@main/dense_2/kernel*
valueB"А      
Э
0main/dense_2/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *&
_class
loc:@main/dense_2/kernel*
valueB
 *    
Е
*main/dense_2/kernel/Adam/Initializer/zerosFill:main/dense_2/kernel/Adam/Initializer/zeros/shape_as_tensor0main/dense_2/kernel/Adam/Initializer/zeros/Const* 
_output_shapes
:
АА*
T0*&
_class
loc:@main/dense_2/kernel*

index_type0
Є
main/dense_2/kernel/Adam
VariableV2*
dtype0* 
_output_shapes
:
АА*
shared_name *&
_class
loc:@main/dense_2/kernel*
	container *
shape:
АА
л
main/dense_2/kernel/Adam/AssignAssignmain/dense_2/kernel/Adam*main/dense_2/kernel/Adam/Initializer/zeros*
validate_shape(* 
_output_shapes
:
АА*
use_locking(*
T0*&
_class
loc:@main/dense_2/kernel
Ц
main/dense_2/kernel/Adam/readIdentitymain/dense_2/kernel/Adam*
T0*&
_class
loc:@main/dense_2/kernel* 
_output_shapes
:
АА
µ
<main/dense_2/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*&
_class
loc:@main/dense_2/kernel*
valueB"А      *
dtype0*
_output_shapes
:
Я
2main/dense_2/kernel/Adam_1/Initializer/zeros/ConstConst*&
_class
loc:@main/dense_2/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Л
,main/dense_2/kernel/Adam_1/Initializer/zerosFill<main/dense_2/kernel/Adam_1/Initializer/zeros/shape_as_tensor2main/dense_2/kernel/Adam_1/Initializer/zeros/Const*
T0*&
_class
loc:@main/dense_2/kernel*

index_type0* 
_output_shapes
:
АА
Ї
main/dense_2/kernel/Adam_1
VariableV2*
shared_name *&
_class
loc:@main/dense_2/kernel*
	container *
shape:
АА*
dtype0* 
_output_shapes
:
АА
с
!main/dense_2/kernel/Adam_1/AssignAssignmain/dense_2/kernel/Adam_1,main/dense_2/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*&
_class
loc:@main/dense_2/kernel*
validate_shape(* 
_output_shapes
:
АА
Ъ
main/dense_2/kernel/Adam_1/readIdentitymain/dense_2/kernel/Adam_1* 
_output_shapes
:
АА*
T0*&
_class
loc:@main/dense_2/kernel
Э
(main/dense_2/bias/Adam/Initializer/zerosConst*$
_class
loc:@main/dense_2/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
™
main/dense_2/bias/Adam
VariableV2*$
_class
loc:@main/dense_2/bias*
	container *
shape:А*
dtype0*
_output_shapes	
:А*
shared_name 
ё
main/dense_2/bias/Adam/AssignAssignmain/dense_2/bias/Adam(main/dense_2/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0*$
_class
loc:@main/dense_2/bias
Л
main/dense_2/bias/Adam/readIdentitymain/dense_2/bias/Adam*
T0*$
_class
loc:@main/dense_2/bias*
_output_shapes	
:А
Я
*main/dense_2/bias/Adam_1/Initializer/zerosConst*$
_class
loc:@main/dense_2/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
ђ
main/dense_2/bias/Adam_1
VariableV2*
dtype0*
_output_shapes	
:А*
shared_name *$
_class
loc:@main/dense_2/bias*
	container *
shape:А
д
main/dense_2/bias/Adam_1/AssignAssignmain/dense_2/bias/Adam_1*main/dense_2/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@main/dense_2/bias*
validate_shape(*
_output_shapes	
:А
П
main/dense_2/bias/Adam_1/readIdentitymain/dense_2/bias/Adam_1*
T0*$
_class
loc:@main/dense_2/bias*
_output_shapes	
:А
≥
:main/dense_3/kernel/Adam/Initializer/zeros/shape_as_tensorConst*&
_class
loc:@main/dense_3/kernel*
valueB"   А   *
dtype0*
_output_shapes
:
Э
0main/dense_3/kernel/Adam/Initializer/zeros/ConstConst*&
_class
loc:@main/dense_3/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Е
*main/dense_3/kernel/Adam/Initializer/zerosFill:main/dense_3/kernel/Adam/Initializer/zeros/shape_as_tensor0main/dense_3/kernel/Adam/Initializer/zeros/Const*
T0*&
_class
loc:@main/dense_3/kernel*

index_type0* 
_output_shapes
:
АА
Є
main/dense_3/kernel/Adam
VariableV2*
shared_name *&
_class
loc:@main/dense_3/kernel*
	container *
shape:
АА*
dtype0* 
_output_shapes
:
АА
л
main/dense_3/kernel/Adam/AssignAssignmain/dense_3/kernel/Adam*main/dense_3/kernel/Adam/Initializer/zeros*
use_locking(*
T0*&
_class
loc:@main/dense_3/kernel*
validate_shape(* 
_output_shapes
:
АА
Ц
main/dense_3/kernel/Adam/readIdentitymain/dense_3/kernel/Adam*
T0*&
_class
loc:@main/dense_3/kernel* 
_output_shapes
:
АА
µ
<main/dense_3/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*&
_class
loc:@main/dense_3/kernel*
valueB"   А   *
dtype0*
_output_shapes
:
Я
2main/dense_3/kernel/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *&
_class
loc:@main/dense_3/kernel*
valueB
 *    
Л
,main/dense_3/kernel/Adam_1/Initializer/zerosFill<main/dense_3/kernel/Adam_1/Initializer/zeros/shape_as_tensor2main/dense_3/kernel/Adam_1/Initializer/zeros/Const*
T0*&
_class
loc:@main/dense_3/kernel*

index_type0* 
_output_shapes
:
АА
Ї
main/dense_3/kernel/Adam_1
VariableV2*&
_class
loc:@main/dense_3/kernel*
	container *
shape:
АА*
dtype0* 
_output_shapes
:
АА*
shared_name 
с
!main/dense_3/kernel/Adam_1/AssignAssignmain/dense_3/kernel/Adam_1,main/dense_3/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*&
_class
loc:@main/dense_3/kernel*
validate_shape(* 
_output_shapes
:
АА
Ъ
main/dense_3/kernel/Adam_1/readIdentitymain/dense_3/kernel/Adam_1* 
_output_shapes
:
АА*
T0*&
_class
loc:@main/dense_3/kernel
Э
(main/dense_3/bias/Adam/Initializer/zerosConst*$
_class
loc:@main/dense_3/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
™
main/dense_3/bias/Adam
VariableV2*
dtype0*
_output_shapes	
:А*
shared_name *$
_class
loc:@main/dense_3/bias*
	container *
shape:А
ё
main/dense_3/bias/Adam/AssignAssignmain/dense_3/bias/Adam(main/dense_3/bias/Adam/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@main/dense_3/bias*
validate_shape(*
_output_shapes	
:А
Л
main/dense_3/bias/Adam/readIdentitymain/dense_3/bias/Adam*
T0*$
_class
loc:@main/dense_3/bias*
_output_shapes	
:А
Я
*main/dense_3/bias/Adam_1/Initializer/zerosConst*$
_class
loc:@main/dense_3/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
ђ
main/dense_3/bias/Adam_1
VariableV2*
shape:А*
dtype0*
_output_shapes	
:А*
shared_name *$
_class
loc:@main/dense_3/bias*
	container 
д
main/dense_3/bias/Adam_1/AssignAssignmain/dense_3/bias/Adam_1*main/dense_3/bias/Adam_1/Initializer/zeros*
T0*$
_class
loc:@main/dense_3/bias*
validate_shape(*
_output_shapes	
:А*
use_locking(
П
main/dense_3/bias/Adam_1/readIdentitymain/dense_3/bias/Adam_1*
_output_shapes	
:А*
T0*$
_class
loc:@main/dense_3/bias
©
*main/dense_4/kernel/Adam/Initializer/zerosConst*&
_class
loc:@main/dense_4/kernel*
valueB	А*    *
dtype0*
_output_shapes
:	А
ґ
main/dense_4/kernel/Adam
VariableV2*
shape:	А*
dtype0*
_output_shapes
:	А*
shared_name *&
_class
loc:@main/dense_4/kernel*
	container 
к
main/dense_4/kernel/Adam/AssignAssignmain/dense_4/kernel/Adam*main/dense_4/kernel/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:	А*
use_locking(*
T0*&
_class
loc:@main/dense_4/kernel
Х
main/dense_4/kernel/Adam/readIdentitymain/dense_4/kernel/Adam*
T0*&
_class
loc:@main/dense_4/kernel*
_output_shapes
:	А
Ђ
,main/dense_4/kernel/Adam_1/Initializer/zerosConst*&
_class
loc:@main/dense_4/kernel*
valueB	А*    *
dtype0*
_output_shapes
:	А
Є
main/dense_4/kernel/Adam_1
VariableV2*&
_class
loc:@main/dense_4/kernel*
	container *
shape:	А*
dtype0*
_output_shapes
:	А*
shared_name 
р
!main/dense_4/kernel/Adam_1/AssignAssignmain/dense_4/kernel/Adam_1,main/dense_4/kernel/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:	А*
use_locking(*
T0*&
_class
loc:@main/dense_4/kernel
Щ
main/dense_4/kernel/Adam_1/readIdentitymain/dense_4/kernel/Adam_1*
T0*&
_class
loc:@main/dense_4/kernel*
_output_shapes
:	А
Ы
(main/dense_4/bias/Adam/Initializer/zerosConst*$
_class
loc:@main/dense_4/bias*
valueB*    *
dtype0*
_output_shapes
:
®
main/dense_4/bias/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name *$
_class
loc:@main/dense_4/bias*
	container *
shape:
Ё
main/dense_4/bias/Adam/AssignAssignmain/dense_4/bias/Adam(main/dense_4/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*$
_class
loc:@main/dense_4/bias
К
main/dense_4/bias/Adam/readIdentitymain/dense_4/bias/Adam*
T0*$
_class
loc:@main/dense_4/bias*
_output_shapes
:
Э
*main/dense_4/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:*$
_class
loc:@main/dense_4/bias*
valueB*    
™
main/dense_4/bias/Adam_1
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *$
_class
loc:@main/dense_4/bias*
	container 
г
main/dense_4/bias/Adam_1/AssignAssignmain/dense_4/bias/Adam_1*main/dense_4/bias/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*$
_class
loc:@main/dense_4/bias
О
main/dense_4/bias/Adam_1/readIdentitymain/dense_4/bias/Adam_1*
T0*$
_class
loc:@main/dense_4/bias*
_output_shapes
:
W
Adam/learning_rateConst*
valueB
 *Ј—8*
dtype0*
_output_shapes
: 
O

Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
O

Adam/beta2Const*
dtype0*
_output_shapes
: *
valueB
 *wЊ?
Q
Adam/epsilonConst*
valueB
 *wћ+2*
dtype0*
_output_shapes
: 
Л
'Adam/update_main/dense/kernel/ApplyAdam	ApplyAdammain/dense/kernelmain/dense/kernel/Adammain/dense/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon;gradients/main/dense/MatMul_grad/tuple/control_dependency_1*
T0*$
_class
loc:@main/dense/kernel*
use_nesterov( *
_output_shapes
:	А*
use_locking( 
ю
%Adam/update_main/dense/bias/ApplyAdam	ApplyAdammain/dense/biasmain/dense/bias/Adammain/dense/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon<gradients/main/dense/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@main/dense/bias*
use_nesterov( *
_output_shapes	
:А
Ч
)Adam/update_main/dense_1/kernel/ApplyAdam	ApplyAdammain/dense_1/kernelmain/dense_1/kernel/Adammain/dense_1/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon=gradients/main/dense_1/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*&
_class
loc:@main/dense_1/kernel*
use_nesterov( *
_output_shapes
:	@А
К
'Adam/update_main/dense_1/bias/ApplyAdam	ApplyAdammain/dense_1/biasmain/dense_1/bias/Adammain/dense_1/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon>gradients/main/dense_1/BiasAdd_grad/tuple/control_dependency_1*
T0*$
_class
loc:@main/dense_1/bias*
use_nesterov( *
_output_shapes	
:А*
use_locking( 
Ш
)Adam/update_main/dense_2/kernel/ApplyAdam	ApplyAdammain/dense_2/kernelmain/dense_2/kernel/Adammain/dense_2/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon=gradients/main/dense_2/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*&
_class
loc:@main/dense_2/kernel*
use_nesterov( * 
_output_shapes
:
АА
К
'Adam/update_main/dense_2/bias/ApplyAdam	ApplyAdammain/dense_2/biasmain/dense_2/bias/Adammain/dense_2/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon>gradients/main/dense_2/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*$
_class
loc:@main/dense_2/bias*
use_nesterov( *
_output_shapes	
:А
Ш
)Adam/update_main/dense_3/kernel/ApplyAdam	ApplyAdammain/dense_3/kernelmain/dense_3/kernel/Adammain/dense_3/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon=gradients/main/dense_3/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*&
_class
loc:@main/dense_3/kernel*
use_nesterov( * 
_output_shapes
:
АА
К
'Adam/update_main/dense_3/bias/ApplyAdam	ApplyAdammain/dense_3/biasmain/dense_3/bias/Adammain/dense_3/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon>gradients/main/dense_3/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes	
:А*
use_locking( *
T0*$
_class
loc:@main/dense_3/bias
Ч
)Adam/update_main/dense_4/kernel/ApplyAdam	ApplyAdammain/dense_4/kernelmain/dense_4/kernel/Adammain/dense_4/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon=gradients/main/dense_4/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*&
_class
loc:@main/dense_4/kernel*
use_nesterov( *
_output_shapes
:	А
Й
'Adam/update_main/dense_4/bias/ApplyAdam	ApplyAdammain/dense_4/biasmain/dense_4/bias/Adammain/dense_4/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon>gradients/main/dense_4/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*$
_class
loc:@main/dense_4/bias*
use_nesterov( *
_output_shapes
:
Ь
Adam/mulMulbeta1_power/read
Adam/beta1&^Adam/update_main/dense/bias/ApplyAdam(^Adam/update_main/dense/kernel/ApplyAdam(^Adam/update_main/dense_1/bias/ApplyAdam*^Adam/update_main/dense_1/kernel/ApplyAdam(^Adam/update_main/dense_2/bias/ApplyAdam*^Adam/update_main/dense_2/kernel/ApplyAdam(^Adam/update_main/dense_3/bias/ApplyAdam*^Adam/update_main/dense_3/kernel/ApplyAdam(^Adam/update_main/dense_4/bias/ApplyAdam*^Adam/update_main/dense_4/kernel/ApplyAdam*
T0*"
_class
loc:@main/dense/bias*
_output_shapes
: 
Ъ
Adam/AssignAssignbeta1_powerAdam/mul*
validate_shape(*
_output_shapes
: *
use_locking( *
T0*"
_class
loc:@main/dense/bias
Ю

Adam/mul_1Mulbeta2_power/read
Adam/beta2&^Adam/update_main/dense/bias/ApplyAdam(^Adam/update_main/dense/kernel/ApplyAdam(^Adam/update_main/dense_1/bias/ApplyAdam*^Adam/update_main/dense_1/kernel/ApplyAdam(^Adam/update_main/dense_2/bias/ApplyAdam*^Adam/update_main/dense_2/kernel/ApplyAdam(^Adam/update_main/dense_3/bias/ApplyAdam*^Adam/update_main/dense_3/kernel/ApplyAdam(^Adam/update_main/dense_4/bias/ApplyAdam*^Adam/update_main/dense_4/kernel/ApplyAdam*
T0*"
_class
loc:@main/dense/bias*
_output_shapes
: 
Ю
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
use_locking( *
T0*"
_class
loc:@main/dense/bias*
validate_shape(*
_output_shapes
: 
‘
AdamNoOp^Adam/Assign^Adam/Assign_1&^Adam/update_main/dense/bias/ApplyAdam(^Adam/update_main/dense/kernel/ApplyAdam(^Adam/update_main/dense_1/bias/ApplyAdam*^Adam/update_main/dense_1/kernel/ApplyAdam(^Adam/update_main/dense_2/bias/ApplyAdam*^Adam/update_main/dense_2/kernel/ApplyAdam(^Adam/update_main/dense_3/bias/ApplyAdam*^Adam/update_main/dense_3/kernel/ApplyAdam(^Adam/update_main/dense_4/bias/ApplyAdam*^Adam/update_main/dense_4/kernel/ApplyAdam
Є
AssignAssigntarget/dense/kernelmain/dense/kernel/read*
validate_shape(*
_output_shapes
:	А*
use_locking(*
T0*&
_class
loc:@target/dense/kernel
∞
Assign_1Assigntarget/dense/biasmain/dense/bias/read*
T0*$
_class
loc:@target/dense/bias*
validate_shape(*
_output_shapes	
:А*
use_locking(
ј
Assign_2Assigntarget/dense_1/kernelmain/dense_1/kernel/read*
use_locking(*
T0*(
_class
loc:@target/dense_1/kernel*
validate_shape(*
_output_shapes
:	@А
ґ
Assign_3Assigntarget/dense_1/biasmain/dense_1/bias/read*
T0*&
_class
loc:@target/dense_1/bias*
validate_shape(*
_output_shapes	
:А*
use_locking(
Ѕ
Assign_4Assigntarget/dense_2/kernelmain/dense_2/kernel/read*
validate_shape(* 
_output_shapes
:
АА*
use_locking(*
T0*(
_class
loc:@target/dense_2/kernel
ґ
Assign_5Assigntarget/dense_2/biasmain/dense_2/bias/read*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0*&
_class
loc:@target/dense_2/bias
Ѕ
Assign_6Assigntarget/dense_3/kernelmain/dense_3/kernel/read*
T0*(
_class
loc:@target/dense_3/kernel*
validate_shape(* 
_output_shapes
:
АА*
use_locking(
ґ
Assign_7Assigntarget/dense_3/biasmain/dense_3/bias/read*
use_locking(*
T0*&
_class
loc:@target/dense_3/bias*
validate_shape(*
_output_shapes	
:А
ј
Assign_8Assigntarget/dense_4/kernelmain/dense_4/kernel/read*
validate_shape(*
_output_shapes
:	А*
use_locking(*
T0*(
_class
loc:@target/dense_4/kernel
µ
Assign_9Assigntarget/dense_4/biasmain/dense_4/bias/read*
use_locking(*
T0*&
_class
loc:@target/dense_4/bias*
validate_shape(*
_output_shapes
:
Т

initNoOp^beta1_power/Assign^beta2_power/Assign^main/dense/bias/Adam/Assign^main/dense/bias/Adam_1/Assign^main/dense/bias/Assign^main/dense/kernel/Adam/Assign ^main/dense/kernel/Adam_1/Assign^main/dense/kernel/Assign^main/dense_1/bias/Adam/Assign ^main/dense_1/bias/Adam_1/Assign^main/dense_1/bias/Assign ^main/dense_1/kernel/Adam/Assign"^main/dense_1/kernel/Adam_1/Assign^main/dense_1/kernel/Assign^main/dense_2/bias/Adam/Assign ^main/dense_2/bias/Adam_1/Assign^main/dense_2/bias/Assign ^main/dense_2/kernel/Adam/Assign"^main/dense_2/kernel/Adam_1/Assign^main/dense_2/kernel/Assign^main/dense_3/bias/Adam/Assign ^main/dense_3/bias/Adam_1/Assign^main/dense_3/bias/Assign ^main/dense_3/kernel/Adam/Assign"^main/dense_3/kernel/Adam_1/Assign^main/dense_3/kernel/Assign^main/dense_4/bias/Adam/Assign ^main/dense_4/bias/Adam_1/Assign^main/dense_4/bias/Assign ^main/dense_4/kernel/Adam/Assign"^main/dense_4/kernel/Adam_1/Assign^main/dense_4/kernel/Assign^target/dense/bias/Assign^target/dense/kernel/Assign^target/dense_1/bias/Assign^target/dense_1/kernel/Assign^target/dense_2/bias/Assign^target/dense_2/kernel/Assign^target/dense_3/bias/Assign^target/dense_3/kernel/Assign^target/dense_4/bias/Assign^target/dense_4/kernel/Assign
R
Placeholder_4Placeholder*
dtype0*
_output_shapes
:*
shape:
R
reward/tagsConst*
dtype0*
_output_shapes
: *
valueB Breward
T
rewardScalarSummaryreward/tagsPlaceholder_4*
T0*
_output_shapes
: 
K
Merge/MergeSummaryMergeSummaryreward*
N*
_output_shapes
: "Џr°ч©     ≠W©M	ЊщƒўЏд÷AJЬЯ
Ш п
,
Abs
x"T
y"T"
Ttype:

2	
:
Add
x"T
y"T
z"T"
Ttype:
2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	АР
о
	ApplyAdam
var"TА	
m"TА	
v"TА
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"TА" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"TА

value"T

output_ref"TА"	
Ttype"
validate_shapebool("
use_lockingbool(Ш
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
,
Cos
x"T
y"T"
Ttype:

2
S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
9
FloorMod
x"T
y"T
z"T"
Ttype:

2	
.
Identity

input"T
output"T"	
Ttype
:
InvertPermutation
x"T
y"T"
Ttype0:
2	
:
Less
x"T
y"T
z
"
Ttype:
2	
?
	LessEqual
x"T
y"T
z
"
Ttype:
2	
p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2
;
Maximum
x"T
y"T
z"T"
Ttype:

2	Р
Н
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
8
MergeSummary
inputs*N
summary"
Nint(0
;
Minimum
x"T
y"T
z"T"
Ttype:

2	Р
=
Mul
x"T
y"T
z"T"
Ttype:
2	Р
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
D
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
P
ScalarSummary
tags
values"T
summary"
Ttype:
2	
?
Select
	condition

t"T
e"T
output"T"	
Ttype
<
Selu
features"T
activations"T"
Ttype:
2
M
SeluGrad
	gradients"T
outputs"T
	backprops"T"
Ttype:
2
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
/
Sign
x"T
y"T"
Ttype:

2	
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
:
Sub
x"T
y"T
z"T"
Ttype:
2	
М
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
P
Unpack

value"T
output"T*num"
numint("	
Ttype"
axisint 
s

VariableV2
ref"dtypeА"
shapeshape"
dtypetype"
	containerstring "
shared_namestring И*	1.9.0-rc12v1.9.0-rc0-35-g17d6639b55џЊ
n
PlaceholderPlaceholder*
shape:€€€€€€€€€*
dtype0*'
_output_shapes
:€€€€€€€€€
p
Placeholder_1Placeholder*
dtype0*'
_output_shapes
:€€€€€€€€€*
shape:€€€€€€€€€
p
Placeholder_2Placeholder*
dtype0*'
_output_shapes
:€€€€€€€€€*
shape:€€€€€€€€€
p
Placeholder_3Placeholder*
shape:€€€€€€€€€*
dtype0*'
_output_shapes
:€€€€€€€€€
d
main/Tile/multiplesConst*
_output_shapes
:*
valueB"      *
dtype0
w
	main/TileTilePlaceholdermain/Tile/multiples*

Tmultiples0*
T0*'
_output_shapes
:€€€€€€€€€ 
c
main/Reshape/shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:
v
main/ReshapeReshape	main/Tilemain/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
©
2main/dense/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*$
_class
loc:@main/dense/kernel*
valueB"   А   
Ы
0main/dense/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *$
_class
loc:@main/dense/kernel*
valueB
 *JQZЊ
Ы
0main/dense/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *$
_class
loc:@main/dense/kernel*
valueB
 *JQZ>
х
:main/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform2main/dense/kernel/Initializer/random_uniform/shape*
T0*$
_class
loc:@main/dense/kernel*
seed2 *
dtype0*
_output_shapes
:	А*

seed 
в
0main/dense/kernel/Initializer/random_uniform/subSub0main/dense/kernel/Initializer/random_uniform/max0main/dense/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*$
_class
loc:@main/dense/kernel
х
0main/dense/kernel/Initializer/random_uniform/mulMul:main/dense/kernel/Initializer/random_uniform/RandomUniform0main/dense/kernel/Initializer/random_uniform/sub*
T0*$
_class
loc:@main/dense/kernel*
_output_shapes
:	А
з
,main/dense/kernel/Initializer/random_uniformAdd0main/dense/kernel/Initializer/random_uniform/mul0main/dense/kernel/Initializer/random_uniform/min*
T0*$
_class
loc:@main/dense/kernel*
_output_shapes
:	А
≠
main/dense/kernel
VariableV2*$
_class
loc:@main/dense/kernel*
	container *
shape:	А*
dtype0*
_output_shapes
:	А*
shared_name 
№
main/dense/kernel/AssignAssignmain/dense/kernel,main/dense/kernel/Initializer/random_uniform*
validate_shape(*
_output_shapes
:	А*
use_locking(*
T0*$
_class
loc:@main/dense/kernel
Е
main/dense/kernel/readIdentitymain/dense/kernel*
T0*$
_class
loc:@main/dense/kernel*
_output_shapes
:	А
Ф
!main/dense/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:А*"
_class
loc:@main/dense/bias*
valueBА*    
°
main/dense/bias
VariableV2*
	container *
shape:А*
dtype0*
_output_shapes	
:А*
shared_name *"
_class
loc:@main/dense/bias
«
main/dense/bias/AssignAssignmain/dense/bias!main/dense/bias/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@main/dense/bias*
validate_shape(*
_output_shapes	
:А
{
main/dense/bias/readIdentitymain/dense/bias*
T0*"
_class
loc:@main/dense/bias*
_output_shapes	
:А
Ъ
main/dense/MatMulMatMulmain/Reshapemain/dense/kernel/read*
transpose_b( *
T0*(
_output_shapes
:€€€€€€€€€А*
transpose_a( 
Р
main/dense/BiasAddBiasAddmain/dense/MatMulmain/dense/bias/read*
data_formatNHWC*(
_output_shapes
:€€€€€€€€€А*
T0
^
main/dense/SeluSelumain/dense/BiasAdd*
T0*(
_output_shapes
:€€€€€€€€€А
e
main/Reshape_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"€€€€   
~
main/Reshape_1ReshapePlaceholder_1main/Reshape_1/shape*'
_output_shapes
:€€€€€€€€€*
T0*
Tshape0
я

main/ConstConst*Ь
valueТBП@"А    џI@џ…@дЋAџIA—S{AдЋЦAянѓAџ…A÷1вA—SыAж:
BдЋBв\#Bян/BЁ~<BџIBЎ†UB÷1bB‘¬nB—S{BgтГBж:КBeГРBдЋЦBcЭBв\£B`•©BянѓB^6ґBЁ~ЉB\«¬Bџ…BYXѕBЎ†’BWйџB÷1вBUzиB‘¬оBRхB—SыB(ќ CgтCІCж:
C&_CeГC•ІCдЋC#рCcCҐ8 Cв\#C!Б&C`•)C†…,Cян/C3C^66CЮZ9CЁ~<C£?C\«BCЫлEC*
dtype0*
_output_shapes

:@
Й
main/MatMulMatMulmain/Reshape_1
main/Const*
T0*'
_output_shapes
:€€€€€€€€€@*
transpose_a( *
transpose_b( 
N
main/CosCosmain/MatMul*
T0*'
_output_shapes
:€€€€€€€€€@
≠
4main/dense_1/kernel/Initializer/random_uniform/shapeConst*&
_class
loc:@main/dense_1/kernel*
valueB"@   А   *
dtype0*
_output_shapes
:
Я
2main/dense_1/kernel/Initializer/random_uniform/minConst*&
_class
loc:@main/dense_1/kernel*
valueB
 *у5Њ*
dtype0*
_output_shapes
: 
Я
2main/dense_1/kernel/Initializer/random_uniform/maxConst*&
_class
loc:@main/dense_1/kernel*
valueB
 *у5>*
dtype0*
_output_shapes
: 
ы
<main/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform4main/dense_1/kernel/Initializer/random_uniform/shape*
T0*&
_class
loc:@main/dense_1/kernel*
seed2 *
dtype0*
_output_shapes
:	@А*

seed 
к
2main/dense_1/kernel/Initializer/random_uniform/subSub2main/dense_1/kernel/Initializer/random_uniform/max2main/dense_1/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@main/dense_1/kernel*
_output_shapes
: 
э
2main/dense_1/kernel/Initializer/random_uniform/mulMul<main/dense_1/kernel/Initializer/random_uniform/RandomUniform2main/dense_1/kernel/Initializer/random_uniform/sub*
T0*&
_class
loc:@main/dense_1/kernel*
_output_shapes
:	@А
п
.main/dense_1/kernel/Initializer/random_uniformAdd2main/dense_1/kernel/Initializer/random_uniform/mul2main/dense_1/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@main/dense_1/kernel*
_output_shapes
:	@А
±
main/dense_1/kernel
VariableV2*
	container *
shape:	@А*
dtype0*
_output_shapes
:	@А*
shared_name *&
_class
loc:@main/dense_1/kernel
д
main/dense_1/kernel/AssignAssignmain/dense_1/kernel.main/dense_1/kernel/Initializer/random_uniform*
validate_shape(*
_output_shapes
:	@А*
use_locking(*
T0*&
_class
loc:@main/dense_1/kernel
Л
main/dense_1/kernel/readIdentitymain/dense_1/kernel*
T0*&
_class
loc:@main/dense_1/kernel*
_output_shapes
:	@А
Ш
#main/dense_1/bias/Initializer/zerosConst*$
_class
loc:@main/dense_1/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
•
main/dense_1/bias
VariableV2*
dtype0*
_output_shapes	
:А*
shared_name *$
_class
loc:@main/dense_1/bias*
	container *
shape:А
ѕ
main/dense_1/bias/AssignAssignmain/dense_1/bias#main/dense_1/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0*$
_class
loc:@main/dense_1/bias
Б
main/dense_1/bias/readIdentitymain/dense_1/bias*
T0*$
_class
loc:@main/dense_1/bias*
_output_shapes	
:А
Ъ
main/dense_1/MatMulMatMulmain/Cosmain/dense_1/kernel/read*
transpose_b( *
T0*(
_output_shapes
:€€€€€€€€€А*
transpose_a( 
Ц
main/dense_1/BiasAddBiasAddmain/dense_1/MatMulmain/dense_1/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:€€€€€€€€€А
b
main/dense_1/ReluRelumain/dense_1/BiasAdd*(
_output_shapes
:€€€€€€€€€А*
T0
f
main/MulMulmain/dense/Selumain/dense_1/Relu*(
_output_shapes
:€€€€€€€€€А*
T0
≠
4main/dense_2/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*&
_class
loc:@main/dense_2/kernel*
valueB"А      
Я
2main/dense_2/kernel/Initializer/random_uniform/minConst*&
_class
loc:@main/dense_2/kernel*
valueB
 *шK∆љ*
dtype0*
_output_shapes
: 
Я
2main/dense_2/kernel/Initializer/random_uniform/maxConst*&
_class
loc:@main/dense_2/kernel*
valueB
 *шK∆=*
dtype0*
_output_shapes
: 
ь
<main/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform4main/dense_2/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
АА*

seed *
T0*&
_class
loc:@main/dense_2/kernel*
seed2 
к
2main/dense_2/kernel/Initializer/random_uniform/subSub2main/dense_2/kernel/Initializer/random_uniform/max2main/dense_2/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@main/dense_2/kernel*
_output_shapes
: 
ю
2main/dense_2/kernel/Initializer/random_uniform/mulMul<main/dense_2/kernel/Initializer/random_uniform/RandomUniform2main/dense_2/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
АА*
T0*&
_class
loc:@main/dense_2/kernel
р
.main/dense_2/kernel/Initializer/random_uniformAdd2main/dense_2/kernel/Initializer/random_uniform/mul2main/dense_2/kernel/Initializer/random_uniform/min* 
_output_shapes
:
АА*
T0*&
_class
loc:@main/dense_2/kernel
≥
main/dense_2/kernel
VariableV2*
dtype0* 
_output_shapes
:
АА*
shared_name *&
_class
loc:@main/dense_2/kernel*
	container *
shape:
АА
е
main/dense_2/kernel/AssignAssignmain/dense_2/kernel.main/dense_2/kernel/Initializer/random_uniform*
T0*&
_class
loc:@main/dense_2/kernel*
validate_shape(* 
_output_shapes
:
АА*
use_locking(
М
main/dense_2/kernel/readIdentitymain/dense_2/kernel*
T0*&
_class
loc:@main/dense_2/kernel* 
_output_shapes
:
АА
Ш
#main/dense_2/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:А*$
_class
loc:@main/dense_2/bias*
valueBА*    
•
main/dense_2/bias
VariableV2*
dtype0*
_output_shapes	
:А*
shared_name *$
_class
loc:@main/dense_2/bias*
	container *
shape:А
ѕ
main/dense_2/bias/AssignAssignmain/dense_2/bias#main/dense_2/bias/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@main/dense_2/bias*
validate_shape(*
_output_shapes	
:А
Б
main/dense_2/bias/readIdentitymain/dense_2/bias*
_output_shapes	
:А*
T0*$
_class
loc:@main/dense_2/bias
Ъ
main/dense_2/MatMulMatMulmain/Mulmain/dense_2/kernel/read*
T0*(
_output_shapes
:€€€€€€€€€А*
transpose_a( *
transpose_b( 
Ц
main/dense_2/BiasAddBiasAddmain/dense_2/MatMulmain/dense_2/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:€€€€€€€€€А
b
main/dense_2/ReluRelumain/dense_2/BiasAdd*
T0*(
_output_shapes
:€€€€€€€€€А
≠
4main/dense_3/kernel/Initializer/random_uniform/shapeConst*&
_class
loc:@main/dense_3/kernel*
valueB"   А   *
dtype0*
_output_shapes
:
Я
2main/dense_3/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *&
_class
loc:@main/dense_3/kernel*
valueB
 *шK∆љ
Я
2main/dense_3/kernel/Initializer/random_uniform/maxConst*&
_class
loc:@main/dense_3/kernel*
valueB
 *шK∆=*
dtype0*
_output_shapes
: 
ь
<main/dense_3/kernel/Initializer/random_uniform/RandomUniformRandomUniform4main/dense_3/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
АА*

seed *
T0*&
_class
loc:@main/dense_3/kernel*
seed2 
к
2main/dense_3/kernel/Initializer/random_uniform/subSub2main/dense_3/kernel/Initializer/random_uniform/max2main/dense_3/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@main/dense_3/kernel*
_output_shapes
: 
ю
2main/dense_3/kernel/Initializer/random_uniform/mulMul<main/dense_3/kernel/Initializer/random_uniform/RandomUniform2main/dense_3/kernel/Initializer/random_uniform/sub*
T0*&
_class
loc:@main/dense_3/kernel* 
_output_shapes
:
АА
р
.main/dense_3/kernel/Initializer/random_uniformAdd2main/dense_3/kernel/Initializer/random_uniform/mul2main/dense_3/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@main/dense_3/kernel* 
_output_shapes
:
АА
≥
main/dense_3/kernel
VariableV2*
dtype0* 
_output_shapes
:
АА*
shared_name *&
_class
loc:@main/dense_3/kernel*
	container *
shape:
АА
е
main/dense_3/kernel/AssignAssignmain/dense_3/kernel.main/dense_3/kernel/Initializer/random_uniform*
validate_shape(* 
_output_shapes
:
АА*
use_locking(*
T0*&
_class
loc:@main/dense_3/kernel
М
main/dense_3/kernel/readIdentitymain/dense_3/kernel*
T0*&
_class
loc:@main/dense_3/kernel* 
_output_shapes
:
АА
Ш
#main/dense_3/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:А*$
_class
loc:@main/dense_3/bias*
valueBА*    
•
main/dense_3/bias
VariableV2*
dtype0*
_output_shapes	
:А*
shared_name *$
_class
loc:@main/dense_3/bias*
	container *
shape:А
ѕ
main/dense_3/bias/AssignAssignmain/dense_3/bias#main/dense_3/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0*$
_class
loc:@main/dense_3/bias
Б
main/dense_3/bias/readIdentitymain/dense_3/bias*
T0*$
_class
loc:@main/dense_3/bias*
_output_shapes	
:А
£
main/dense_3/MatMulMatMulmain/dense_2/Relumain/dense_3/kernel/read*
T0*(
_output_shapes
:€€€€€€€€€А*
transpose_a( *
transpose_b( 
Ц
main/dense_3/BiasAddBiasAddmain/dense_3/MatMulmain/dense_3/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:€€€€€€€€€А
b
main/dense_3/ReluRelumain/dense_3/BiasAdd*(
_output_shapes
:€€€€€€€€€А*
T0
≠
4main/dense_4/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*&
_class
loc:@main/dense_4/kernel*
valueB"А      
Я
2main/dense_4/kernel/Initializer/random_uniform/minConst*&
_class
loc:@main/dense_4/kernel*
valueB
 *Сэ[Њ*
dtype0*
_output_shapes
: 
Я
2main/dense_4/kernel/Initializer/random_uniform/maxConst*&
_class
loc:@main/dense_4/kernel*
valueB
 *Сэ[>*
dtype0*
_output_shapes
: 
ы
<main/dense_4/kernel/Initializer/random_uniform/RandomUniformRandomUniform4main/dense_4/kernel/Initializer/random_uniform/shape*

seed *
T0*&
_class
loc:@main/dense_4/kernel*
seed2 *
dtype0*
_output_shapes
:	А
к
2main/dense_4/kernel/Initializer/random_uniform/subSub2main/dense_4/kernel/Initializer/random_uniform/max2main/dense_4/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*&
_class
loc:@main/dense_4/kernel
э
2main/dense_4/kernel/Initializer/random_uniform/mulMul<main/dense_4/kernel/Initializer/random_uniform/RandomUniform2main/dense_4/kernel/Initializer/random_uniform/sub*
T0*&
_class
loc:@main/dense_4/kernel*
_output_shapes
:	А
п
.main/dense_4/kernel/Initializer/random_uniformAdd2main/dense_4/kernel/Initializer/random_uniform/mul2main/dense_4/kernel/Initializer/random_uniform/min*
_output_shapes
:	А*
T0*&
_class
loc:@main/dense_4/kernel
±
main/dense_4/kernel
VariableV2*
dtype0*
_output_shapes
:	А*
shared_name *&
_class
loc:@main/dense_4/kernel*
	container *
shape:	А
д
main/dense_4/kernel/AssignAssignmain/dense_4/kernel.main/dense_4/kernel/Initializer/random_uniform*
use_locking(*
T0*&
_class
loc:@main/dense_4/kernel*
validate_shape(*
_output_shapes
:	А
Л
main/dense_4/kernel/readIdentitymain/dense_4/kernel*
T0*&
_class
loc:@main/dense_4/kernel*
_output_shapes
:	А
Ц
#main/dense_4/bias/Initializer/zerosConst*$
_class
loc:@main/dense_4/bias*
valueB*    *
dtype0*
_output_shapes
:
£
main/dense_4/bias
VariableV2*
shared_name *$
_class
loc:@main/dense_4/bias*
	container *
shape:*
dtype0*
_output_shapes
:
ќ
main/dense_4/bias/AssignAssignmain/dense_4/bias#main/dense_4/bias/Initializer/zeros*
T0*$
_class
loc:@main/dense_4/bias*
validate_shape(*
_output_shapes
:*
use_locking(
А
main/dense_4/bias/readIdentitymain/dense_4/bias*
T0*$
_class
loc:@main/dense_4/bias*
_output_shapes
:
Ґ
main/dense_4/MatMulMatMulmain/dense_3/Relumain/dense_4/kernel/read*'
_output_shapes
:€€€€€€€€€*
transpose_a( *
transpose_b( *
T0
Х
main/dense_4/BiasAddBiasAddmain/dense_4/MatMulmain/dense_4/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€
N
main/Const_1Const*
value	B : *
dtype0*
_output_shapes
: 
V
main/split/split_dimConst*
value	B : *
dtype0*
_output_shapes
: 
“

main/splitSplitmain/split/split_dimmain/dense_4/BiasAdd*
T0*ц
_output_shapesг
а:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€*
	num_split 
Ј
main/transpose/xPack
main/splitmain/split:1main/split:2main/split:3main/split:4main/split:5main/split:6main/split:7main/split:8main/split:9main/split:10main/split:11main/split:12main/split:13main/split:14main/split:15main/split:16main/split:17main/split:18main/split:19main/split:20main/split:21main/split:22main/split:23main/split:24main/split:25main/split:26main/split:27main/split:28main/split:29main/split:30main/split:31*
N *+
_output_shapes
: €€€€€€€€€*
T0*

axis 
h
main/transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:
Е
main/transpose	Transposemain/transpose/xmain/transpose/perm*
T0*+
_output_shapes
: €€€€€€€€€*
Tperm0
f
target/Tile/multiplesConst*
valueB"      *
dtype0*
_output_shapes
:
{
target/TileTilePlaceholdertarget/Tile/multiples*'
_output_shapes
:€€€€€€€€€ *

Tmultiples0*
T0
e
target/Reshape/shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:
|
target/ReshapeReshapetarget/Tiletarget/Reshape/shape*'
_output_shapes
:€€€€€€€€€*
T0*
Tshape0
≠
4target/dense/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*&
_class
loc:@target/dense/kernel*
valueB"   А   
Я
2target/dense/kernel/Initializer/random_uniform/minConst*&
_class
loc:@target/dense/kernel*
valueB
 *JQZЊ*
dtype0*
_output_shapes
: 
Я
2target/dense/kernel/Initializer/random_uniform/maxConst*&
_class
loc:@target/dense/kernel*
valueB
 *JQZ>*
dtype0*
_output_shapes
: 
ы
<target/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform4target/dense/kernel/Initializer/random_uniform/shape*
seed2 *
dtype0*
_output_shapes
:	А*

seed *
T0*&
_class
loc:@target/dense/kernel
к
2target/dense/kernel/Initializer/random_uniform/subSub2target/dense/kernel/Initializer/random_uniform/max2target/dense/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@target/dense/kernel*
_output_shapes
: 
э
2target/dense/kernel/Initializer/random_uniform/mulMul<target/dense/kernel/Initializer/random_uniform/RandomUniform2target/dense/kernel/Initializer/random_uniform/sub*
T0*&
_class
loc:@target/dense/kernel*
_output_shapes
:	А
п
.target/dense/kernel/Initializer/random_uniformAdd2target/dense/kernel/Initializer/random_uniform/mul2target/dense/kernel/Initializer/random_uniform/min*
_output_shapes
:	А*
T0*&
_class
loc:@target/dense/kernel
±
target/dense/kernel
VariableV2*
	container *
shape:	А*
dtype0*
_output_shapes
:	А*
shared_name *&
_class
loc:@target/dense/kernel
д
target/dense/kernel/AssignAssigntarget/dense/kernel.target/dense/kernel/Initializer/random_uniform*
use_locking(*
T0*&
_class
loc:@target/dense/kernel*
validate_shape(*
_output_shapes
:	А
Л
target/dense/kernel/readIdentitytarget/dense/kernel*
T0*&
_class
loc:@target/dense/kernel*
_output_shapes
:	А
Ш
#target/dense/bias/Initializer/zerosConst*$
_class
loc:@target/dense/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
•
target/dense/bias
VariableV2*$
_class
loc:@target/dense/bias*
	container *
shape:А*
dtype0*
_output_shapes	
:А*
shared_name 
ѕ
target/dense/bias/AssignAssigntarget/dense/bias#target/dense/bias/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@target/dense/bias*
validate_shape(*
_output_shapes	
:А
Б
target/dense/bias/readIdentitytarget/dense/bias*
_output_shapes	
:А*
T0*$
_class
loc:@target/dense/bias
†
target/dense/MatMulMatMultarget/Reshapetarget/dense/kernel/read*
T0*(
_output_shapes
:€€€€€€€€€А*
transpose_a( *
transpose_b( 
Ц
target/dense/BiasAddBiasAddtarget/dense/MatMultarget/dense/bias/read*
data_formatNHWC*(
_output_shapes
:€€€€€€€€€А*
T0
b
target/dense/SeluSelutarget/dense/BiasAdd*(
_output_shapes
:€€€€€€€€€А*
T0
g
target/Reshape_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"€€€€   
В
target/Reshape_1ReshapePlaceholder_1target/Reshape_1/shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
б
target/ConstConst*
dtype0*
_output_shapes

:@*Ь
valueТBП@"А    џI@џ…@дЋAџIA—S{AдЋЦAянѓAџ…A÷1вA—SыAж:
BдЋBв\#Bян/BЁ~<BџIBЎ†UB÷1bB‘¬nB—S{BgтГBж:КBeГРBдЋЦBcЭBв\£B`•©BянѓB^6ґBЁ~ЉB\«¬Bџ…BYXѕBЎ†’BWйџB÷1вBUzиB‘¬оBRхB—SыB(ќ CgтCІCж:
C&_CeГC•ІCдЋC#рCcCҐ8 Cв\#C!Б&C`•)C†…,Cян/C3C^66CЮZ9CЁ~<C£?C\«BCЫлEC
П
target/MatMulMatMultarget/Reshape_1target/Const*
T0*'
_output_shapes
:€€€€€€€€€@*
transpose_a( *
transpose_b( 
R

target/CosCostarget/MatMul*
T0*'
_output_shapes
:€€€€€€€€€@
±
6target/dense_1/kernel/Initializer/random_uniform/shapeConst*(
_class
loc:@target/dense_1/kernel*
valueB"@   А   *
dtype0*
_output_shapes
:
£
4target/dense_1/kernel/Initializer/random_uniform/minConst*(
_class
loc:@target/dense_1/kernel*
valueB
 *у5Њ*
dtype0*
_output_shapes
: 
£
4target/dense_1/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *(
_class
loc:@target/dense_1/kernel*
valueB
 *у5>
Б
>target/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform6target/dense_1/kernel/Initializer/random_uniform/shape*

seed *
T0*(
_class
loc:@target/dense_1/kernel*
seed2 *
dtype0*
_output_shapes
:	@А
т
4target/dense_1/kernel/Initializer/random_uniform/subSub4target/dense_1/kernel/Initializer/random_uniform/max4target/dense_1/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@target/dense_1/kernel*
_output_shapes
: 
Е
4target/dense_1/kernel/Initializer/random_uniform/mulMul>target/dense_1/kernel/Initializer/random_uniform/RandomUniform4target/dense_1/kernel/Initializer/random_uniform/sub*
T0*(
_class
loc:@target/dense_1/kernel*
_output_shapes
:	@А
ч
0target/dense_1/kernel/Initializer/random_uniformAdd4target/dense_1/kernel/Initializer/random_uniform/mul4target/dense_1/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@target/dense_1/kernel*
_output_shapes
:	@А
µ
target/dense_1/kernel
VariableV2*
shared_name *(
_class
loc:@target/dense_1/kernel*
	container *
shape:	@А*
dtype0*
_output_shapes
:	@А
м
target/dense_1/kernel/AssignAssigntarget/dense_1/kernel0target/dense_1/kernel/Initializer/random_uniform*
validate_shape(*
_output_shapes
:	@А*
use_locking(*
T0*(
_class
loc:@target/dense_1/kernel
С
target/dense_1/kernel/readIdentitytarget/dense_1/kernel*
T0*(
_class
loc:@target/dense_1/kernel*
_output_shapes
:	@А
Ь
%target/dense_1/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:А*&
_class
loc:@target/dense_1/bias*
valueBА*    
©
target/dense_1/bias
VariableV2*
dtype0*
_output_shapes	
:А*
shared_name *&
_class
loc:@target/dense_1/bias*
	container *
shape:А
„
target/dense_1/bias/AssignAssigntarget/dense_1/bias%target/dense_1/bias/Initializer/zeros*
T0*&
_class
loc:@target/dense_1/bias*
validate_shape(*
_output_shapes	
:А*
use_locking(
З
target/dense_1/bias/readIdentitytarget/dense_1/bias*
T0*&
_class
loc:@target/dense_1/bias*
_output_shapes	
:А
†
target/dense_1/MatMulMatMul
target/Costarget/dense_1/kernel/read*(
_output_shapes
:€€€€€€€€€А*
transpose_a( *
transpose_b( *
T0
Ь
target/dense_1/BiasAddBiasAddtarget/dense_1/MatMultarget/dense_1/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:€€€€€€€€€А
f
target/dense_1/ReluRelutarget/dense_1/BiasAdd*
T0*(
_output_shapes
:€€€€€€€€€А
l

target/MulMultarget/dense/Selutarget/dense_1/Relu*
T0*(
_output_shapes
:€€€€€€€€€А
±
6target/dense_2/kernel/Initializer/random_uniform/shapeConst*(
_class
loc:@target/dense_2/kernel*
valueB"А      *
dtype0*
_output_shapes
:
£
4target/dense_2/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *(
_class
loc:@target/dense_2/kernel*
valueB
 *шK∆љ
£
4target/dense_2/kernel/Initializer/random_uniform/maxConst*(
_class
loc:@target/dense_2/kernel*
valueB
 *шK∆=*
dtype0*
_output_shapes
: 
В
>target/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform6target/dense_2/kernel/Initializer/random_uniform/shape*

seed *
T0*(
_class
loc:@target/dense_2/kernel*
seed2 *
dtype0* 
_output_shapes
:
АА
т
4target/dense_2/kernel/Initializer/random_uniform/subSub4target/dense_2/kernel/Initializer/random_uniform/max4target/dense_2/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@target/dense_2/kernel*
_output_shapes
: 
Ж
4target/dense_2/kernel/Initializer/random_uniform/mulMul>target/dense_2/kernel/Initializer/random_uniform/RandomUniform4target/dense_2/kernel/Initializer/random_uniform/sub*
T0*(
_class
loc:@target/dense_2/kernel* 
_output_shapes
:
АА
ш
0target/dense_2/kernel/Initializer/random_uniformAdd4target/dense_2/kernel/Initializer/random_uniform/mul4target/dense_2/kernel/Initializer/random_uniform/min* 
_output_shapes
:
АА*
T0*(
_class
loc:@target/dense_2/kernel
Ј
target/dense_2/kernel
VariableV2*
dtype0* 
_output_shapes
:
АА*
shared_name *(
_class
loc:@target/dense_2/kernel*
	container *
shape:
АА
н
target/dense_2/kernel/AssignAssigntarget/dense_2/kernel0target/dense_2/kernel/Initializer/random_uniform*
T0*(
_class
loc:@target/dense_2/kernel*
validate_shape(* 
_output_shapes
:
АА*
use_locking(
Т
target/dense_2/kernel/readIdentitytarget/dense_2/kernel* 
_output_shapes
:
АА*
T0*(
_class
loc:@target/dense_2/kernel
Ь
%target/dense_2/bias/Initializer/zerosConst*&
_class
loc:@target/dense_2/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
©
target/dense_2/bias
VariableV2*&
_class
loc:@target/dense_2/bias*
	container *
shape:А*
dtype0*
_output_shapes	
:А*
shared_name 
„
target/dense_2/bias/AssignAssigntarget/dense_2/bias%target/dense_2/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0*&
_class
loc:@target/dense_2/bias
З
target/dense_2/bias/readIdentitytarget/dense_2/bias*
T0*&
_class
loc:@target/dense_2/bias*
_output_shapes	
:А
†
target/dense_2/MatMulMatMul
target/Multarget/dense_2/kernel/read*
transpose_b( *
T0*(
_output_shapes
:€€€€€€€€€А*
transpose_a( 
Ь
target/dense_2/BiasAddBiasAddtarget/dense_2/MatMultarget/dense_2/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:€€€€€€€€€А
f
target/dense_2/ReluRelutarget/dense_2/BiasAdd*(
_output_shapes
:€€€€€€€€€А*
T0
±
6target/dense_3/kernel/Initializer/random_uniform/shapeConst*(
_class
loc:@target/dense_3/kernel*
valueB"   А   *
dtype0*
_output_shapes
:
£
4target/dense_3/kernel/Initializer/random_uniform/minConst*(
_class
loc:@target/dense_3/kernel*
valueB
 *шK∆љ*
dtype0*
_output_shapes
: 
£
4target/dense_3/kernel/Initializer/random_uniform/maxConst*(
_class
loc:@target/dense_3/kernel*
valueB
 *шK∆=*
dtype0*
_output_shapes
: 
В
>target/dense_3/kernel/Initializer/random_uniform/RandomUniformRandomUniform6target/dense_3/kernel/Initializer/random_uniform/shape*
seed2 *
dtype0* 
_output_shapes
:
АА*

seed *
T0*(
_class
loc:@target/dense_3/kernel
т
4target/dense_3/kernel/Initializer/random_uniform/subSub4target/dense_3/kernel/Initializer/random_uniform/max4target/dense_3/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*(
_class
loc:@target/dense_3/kernel
Ж
4target/dense_3/kernel/Initializer/random_uniform/mulMul>target/dense_3/kernel/Initializer/random_uniform/RandomUniform4target/dense_3/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
АА*
T0*(
_class
loc:@target/dense_3/kernel
ш
0target/dense_3/kernel/Initializer/random_uniformAdd4target/dense_3/kernel/Initializer/random_uniform/mul4target/dense_3/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@target/dense_3/kernel* 
_output_shapes
:
АА
Ј
target/dense_3/kernel
VariableV2*
shape:
АА*
dtype0* 
_output_shapes
:
АА*
shared_name *(
_class
loc:@target/dense_3/kernel*
	container 
н
target/dense_3/kernel/AssignAssigntarget/dense_3/kernel0target/dense_3/kernel/Initializer/random_uniform*
T0*(
_class
loc:@target/dense_3/kernel*
validate_shape(* 
_output_shapes
:
АА*
use_locking(
Т
target/dense_3/kernel/readIdentitytarget/dense_3/kernel*
T0*(
_class
loc:@target/dense_3/kernel* 
_output_shapes
:
АА
Ь
%target/dense_3/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:А*&
_class
loc:@target/dense_3/bias*
valueBА*    
©
target/dense_3/bias
VariableV2*
_output_shapes	
:А*
shared_name *&
_class
loc:@target/dense_3/bias*
	container *
shape:А*
dtype0
„
target/dense_3/bias/AssignAssigntarget/dense_3/bias%target/dense_3/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0*&
_class
loc:@target/dense_3/bias
З
target/dense_3/bias/readIdentitytarget/dense_3/bias*
T0*&
_class
loc:@target/dense_3/bias*
_output_shapes	
:А
©
target/dense_3/MatMulMatMultarget/dense_2/Relutarget/dense_3/kernel/read*(
_output_shapes
:€€€€€€€€€А*
transpose_a( *
transpose_b( *
T0
Ь
target/dense_3/BiasAddBiasAddtarget/dense_3/MatMultarget/dense_3/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:€€€€€€€€€А
f
target/dense_3/ReluRelutarget/dense_3/BiasAdd*
T0*(
_output_shapes
:€€€€€€€€€А
±
6target/dense_4/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*(
_class
loc:@target/dense_4/kernel*
valueB"А      *
dtype0
£
4target/dense_4/kernel/Initializer/random_uniform/minConst*(
_class
loc:@target/dense_4/kernel*
valueB
 *Сэ[Њ*
dtype0*
_output_shapes
: 
£
4target/dense_4/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *(
_class
loc:@target/dense_4/kernel*
valueB
 *Сэ[>*
dtype0
Б
>target/dense_4/kernel/Initializer/random_uniform/RandomUniformRandomUniform6target/dense_4/kernel/Initializer/random_uniform/shape*

seed *
T0*(
_class
loc:@target/dense_4/kernel*
seed2 *
dtype0*
_output_shapes
:	А
т
4target/dense_4/kernel/Initializer/random_uniform/subSub4target/dense_4/kernel/Initializer/random_uniform/max4target/dense_4/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*(
_class
loc:@target/dense_4/kernel
Е
4target/dense_4/kernel/Initializer/random_uniform/mulMul>target/dense_4/kernel/Initializer/random_uniform/RandomUniform4target/dense_4/kernel/Initializer/random_uniform/sub*
T0*(
_class
loc:@target/dense_4/kernel*
_output_shapes
:	А
ч
0target/dense_4/kernel/Initializer/random_uniformAdd4target/dense_4/kernel/Initializer/random_uniform/mul4target/dense_4/kernel/Initializer/random_uniform/min*
_output_shapes
:	А*
T0*(
_class
loc:@target/dense_4/kernel
µ
target/dense_4/kernel
VariableV2*
shape:	А*
dtype0*
_output_shapes
:	А*
shared_name *(
_class
loc:@target/dense_4/kernel*
	container 
м
target/dense_4/kernel/AssignAssigntarget/dense_4/kernel0target/dense_4/kernel/Initializer/random_uniform*(
_class
loc:@target/dense_4/kernel*
validate_shape(*
_output_shapes
:	А*
use_locking(*
T0
С
target/dense_4/kernel/readIdentitytarget/dense_4/kernel*
T0*(
_class
loc:@target/dense_4/kernel*
_output_shapes
:	А
Ъ
%target/dense_4/bias/Initializer/zerosConst*&
_class
loc:@target/dense_4/bias*
valueB*    *
dtype0*
_output_shapes
:
І
target/dense_4/bias
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *&
_class
loc:@target/dense_4/bias
÷
target/dense_4/bias/AssignAssigntarget/dense_4/bias%target/dense_4/bias/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*&
_class
loc:@target/dense_4/bias*
validate_shape(
Ж
target/dense_4/bias/readIdentitytarget/dense_4/bias*
T0*&
_class
loc:@target/dense_4/bias*
_output_shapes
:
®
target/dense_4/MatMulMatMultarget/dense_3/Relutarget/dense_4/kernel/read*'
_output_shapes
:€€€€€€€€€*
transpose_a( *
transpose_b( *
T0
Ы
target/dense_4/BiasAddBiasAddtarget/dense_4/MatMultarget/dense_4/bias/read*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€*
T0
P
target/Const_1Const*
value	B : *
dtype0*
_output_shapes
: 
X
target/split/split_dimConst*
value	B : *
dtype0*
_output_shapes
: 
Ў
target/splitSplittarget/split/split_dimtarget/dense_4/BiasAdd*ц
_output_shapesг
а:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€*
	num_split *
T0
щ
target/transpose/xPacktarget/splittarget/split:1target/split:2target/split:3target/split:4target/split:5target/split:6target/split:7target/split:8target/split:9target/split:10target/split:11target/split:12target/split:13target/split:14target/split:15target/split:16target/split:17target/split:18target/split:19target/split:20target/split:21target/split:22target/split:23target/split:24target/split:25target/split:26target/split:27target/split:28target/split:29target/split:30target/split:31*
T0*

axis *
N *+
_output_shapes
: €€€€€€€€€
j
target/transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:
Л
target/transpose	Transposetarget/transpose/xtarget/transpose/perm*
T0*+
_output_shapes
: €€€€€€€€€*
Tperm0
Y
ExpandDims/dimConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: 
y

ExpandDims
ExpandDimsPlaceholder_3ExpandDims/dim*+
_output_shapes
:€€€€€€€€€*

Tdim0*
T0
\
mulMulmain/transpose
ExpandDims*
T0*+
_output_shapes
: €€€€€€€€€
W
Sum/reduction_indicesConst*
dtype0*
_output_shapes
: *
value	B :
u
SumSummulSum/reduction_indices*'
_output_shapes
: €€€€€€€€€*
	keep_dims( *

Tidx0*
T0
R
huber_loss/SubSubSumPlaceholder_2*
T0*
_output_shapes

: 
N
huber_loss/AbsAbshuber_loss/Sub*
T0*
_output_shapes

: 
Y
huber_loss/Minimum/yConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
l
huber_loss/MinimumMinimumhuber_loss/Abshuber_loss/Minimum/y*
T0*
_output_shapes

: 
d
huber_loss/Sub_1Subhuber_loss/Abshuber_loss/Minimum*
_output_shapes

: *
T0
U
huber_loss/ConstConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
f
huber_loss/MulMulhuber_loss/Minimumhuber_loss/Minimum*
_output_shapes

: *
T0
b
huber_loss/Mul_1Mulhuber_loss/Consthuber_loss/Mul*
T0*
_output_shapes

: 
W
huber_loss/Mul_2/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
f
huber_loss/Mul_2Mulhuber_loss/Mul_2/xhuber_loss/Sub_1*
_output_shapes

: *
T0
b
huber_loss/AddAddhuber_loss/Mul_1huber_loss/Mul_2*
_output_shapes

: *
T0
l
'huber_loss/assert_broadcastable/weightsConst*
_output_shapes
: *
valueB
 *  А?*
dtype0
p
-huber_loss/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 
n
,huber_loss/assert_broadcastable/weights/rankConst*
dtype0*
_output_shapes
: *
value	B : 
}
,huber_loss/assert_broadcastable/values/shapeConst*
valueB"       *
dtype0*
_output_shapes
:
m
+huber_loss/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
C
;huber_loss/assert_broadcastable/static_scalar_check_successNoOp
Щ
huber_loss/ToFloat_3/xConst<^huber_loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
valueB
 *  А?*
dtype0
h
huber_loss/Mul_3Mulhuber_loss/Addhuber_loss/ToFloat_3/x*
T0*
_output_shapes

: 
J
sub/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
R
subSubsub/xPlaceholder_1*'
_output_shapes
:€€€€€€€€€*
T0
I
sub_1SubPlaceholder_2Sum*
_output_shapes

: *
T0
K
Less/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
D
LessLesssub_1Less/y*
_output_shapes

: *
T0
L
mul_1Mulsubhuber_loss/Mul_3*
T0*
_output_shapes

: 
V
mul_2MulPlaceholder_1huber_loss/Mul_3*
T0*
_output_shapes

: 
M
SelectSelectLessmul_1mul_2*
_output_shapes

: *
T0
Y
Sum_1/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
o
Sum_1SumSelectSum_1/reduction_indices*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
O
ConstConst*
valueB: *
dtype0*
_output_shapes
:
X
MeanMeanSum_1Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
X
gradients/grad_ys_0Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
k
!gradients/Mean_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
М
gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
c
gradients/Mean_grad/ConstConst*
_output_shapes
:*
valueB: *
dtype0
П
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Const*
_output_shapes
: *

Tmultiples0*
T0
`
gradients/Mean_grad/Const_1Const*
valueB
 *   B*
dtype0*
_output_shapes
: 
В
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Const_1*
T0*
_output_shapes
: 
k
gradients/Sum_1_grad/ShapeConst*
valueB"       *
dtype0*
_output_shapes
:
К
gradients/Sum_1_grad/SizeConst*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
£
gradients/Sum_1_grad/addAddSum_1/reduction_indicesgradients/Sum_1_grad/Size*
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
_output_shapes
: 
©
gradients/Sum_1_grad/modFloorModgradients/Sum_1_grad/addgradients/Sum_1_grad/Size*
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
_output_shapes
: 
О
gradients/Sum_1_grad/Shape_1Const*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
valueB *
dtype0*
_output_shapes
: 
С
 gradients/Sum_1_grad/range/startConst*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
value	B : *
dtype0*
_output_shapes
: 
С
 gradients/Sum_1_grad/range/deltaConst*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
ў
gradients/Sum_1_grad/rangeRange gradients/Sum_1_grad/range/startgradients/Sum_1_grad/Size gradients/Sum_1_grad/range/delta*

Tidx0*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
_output_shapes
:
Р
gradients/Sum_1_grad/Fill/valueConst*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
¬
gradients/Sum_1_grad/FillFillgradients/Sum_1_grad/Shape_1gradients/Sum_1_grad/Fill/value*
_output_shapes
: *
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape*

index_type0
Ж
"gradients/Sum_1_grad/DynamicStitchDynamicStitchgradients/Sum_1_grad/rangegradients/Sum_1_grad/modgradients/Sum_1_grad/Shapegradients/Sum_1_grad/Fill*#
_output_shapes
:€€€€€€€€€*
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
N
П
gradients/Sum_1_grad/Maximum/yConst*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
»
gradients/Sum_1_grad/MaximumMaximum"gradients/Sum_1_grad/DynamicStitchgradients/Sum_1_grad/Maximum/y*
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape*#
_output_shapes
:€€€€€€€€€
Ј
gradients/Sum_1_grad/floordivFloorDivgradients/Sum_1_grad/Shapegradients/Sum_1_grad/Maximum*
_output_shapes
:*
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape
Щ
gradients/Sum_1_grad/ReshapeReshapegradients/Mean_grad/truediv"gradients/Sum_1_grad/DynamicStitch*
_output_shapes
:*
T0*
Tshape0
Щ
gradients/Sum_1_grad/TileTilegradients/Sum_1_grad/Reshapegradients/Sum_1_grad/floordiv*

Tmultiples0*
T0*
_output_shapes

: 
u
 gradients/Select_grad/zeros_likeConst*
_output_shapes

: *
valueB *    *
dtype0
Т
gradients/Select_grad/SelectSelectLessgradients/Sum_1_grad/Tile gradients/Select_grad/zeros_like*
_output_shapes

: *
T0
Ф
gradients/Select_grad/Select_1SelectLess gradients/Select_grad/zeros_likegradients/Sum_1_grad/Tile*
_output_shapes

: *
T0
n
&gradients/Select_grad/tuple/group_depsNoOp^gradients/Select_grad/Select^gradients/Select_grad/Select_1
џ
.gradients/Select_grad/tuple/control_dependencyIdentitygradients/Select_grad/Select'^gradients/Select_grad/tuple/group_deps*/
_class%
#!loc:@gradients/Select_grad/Select*
_output_shapes

: *
T0
б
0gradients/Select_grad/tuple/control_dependency_1Identitygradients/Select_grad/Select_1'^gradients/Select_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/Select_grad/Select_1*
_output_shapes

: 
]
gradients/mul_1_grad/ShapeShapesub*
T0*
out_type0*
_output_shapes
:
m
gradients/mul_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB"       
Ї
*gradients/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_1_grad/Shapegradients/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
К
gradients/mul_1_grad/MulMul.gradients/Select_grad/tuple/control_dependencyhuber_loss/Mul_3*
T0*
_output_shapes

: 
•
gradients/mul_1_grad/SumSumgradients/mul_1_grad/Mul*gradients/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Э
gradients/mul_1_grad/ReshapeReshapegradients/mul_1_grad/Sumgradients/mul_1_grad/Shape*'
_output_shapes
:€€€€€€€€€*
T0*
Tshape0

gradients/mul_1_grad/Mul_1Mulsub.gradients/Select_grad/tuple/control_dependency*
T0*
_output_shapes

: 
Ђ
gradients/mul_1_grad/Sum_1Sumgradients/mul_1_grad/Mul_1,gradients/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ъ
gradients/mul_1_grad/Reshape_1Reshapegradients/mul_1_grad/Sum_1gradients/mul_1_grad/Shape_1*
_output_shapes

: *
T0*
Tshape0
m
%gradients/mul_1_grad/tuple/group_depsNoOp^gradients/mul_1_grad/Reshape^gradients/mul_1_grad/Reshape_1
в
-gradients/mul_1_grad/tuple/control_dependencyIdentitygradients/mul_1_grad/Reshape&^gradients/mul_1_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/mul_1_grad/Reshape*'
_output_shapes
:€€€€€€€€€
я
/gradients/mul_1_grad/tuple/control_dependency_1Identitygradients/mul_1_grad/Reshape_1&^gradients/mul_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/mul_1_grad/Reshape_1*
_output_shapes

: 
g
gradients/mul_2_grad/ShapeShapePlaceholder_1*
T0*
out_type0*
_output_shapes
:
m
gradients/mul_2_grad/Shape_1Const*
valueB"       *
dtype0*
_output_shapes
:
Ї
*gradients/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_2_grad/Shapegradients/mul_2_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
М
gradients/mul_2_grad/MulMul0gradients/Select_grad/tuple/control_dependency_1huber_loss/Mul_3*
T0*
_output_shapes

: 
•
gradients/mul_2_grad/SumSumgradients/mul_2_grad/Mul*gradients/mul_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Э
gradients/mul_2_grad/ReshapeReshapegradients/mul_2_grad/Sumgradients/mul_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
Л
gradients/mul_2_grad/Mul_1MulPlaceholder_10gradients/Select_grad/tuple/control_dependency_1*
T0*
_output_shapes

: 
Ђ
gradients/mul_2_grad/Sum_1Sumgradients/mul_2_grad/Mul_1,gradients/mul_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ъ
gradients/mul_2_grad/Reshape_1Reshapegradients/mul_2_grad/Sum_1gradients/mul_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes

: 
m
%gradients/mul_2_grad/tuple/group_depsNoOp^gradients/mul_2_grad/Reshape^gradients/mul_2_grad/Reshape_1
в
-gradients/mul_2_grad/tuple/control_dependencyIdentitygradients/mul_2_grad/Reshape&^gradients/mul_2_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/mul_2_grad/Reshape*'
_output_shapes
:€€€€€€€€€
я
/gradients/mul_2_grad/tuple/control_dependency_1Identitygradients/mul_2_grad/Reshape_1&^gradients/mul_2_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/mul_2_grad/Reshape_1*
_output_shapes

: 
Ё
gradients/AddNAddN/gradients/mul_1_grad/tuple/control_dependency_1/gradients/mul_2_grad/tuple/control_dependency_1*
T0*1
_class'
%#loc:@gradients/mul_1_grad/Reshape_1*
N*
_output_shapes

: 
v
%gradients/huber_loss/Mul_3_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"       
j
'gradients/huber_loss/Mul_3_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
џ
5gradients/huber_loss/Mul_3_grad/BroadcastGradientArgsBroadcastGradientArgs%gradients/huber_loss/Mul_3_grad/Shape'gradients/huber_loss/Mul_3_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
{
#gradients/huber_loss/Mul_3_grad/MulMulgradients/AddNhuber_loss/ToFloat_3/x*
T0*
_output_shapes

: 
∆
#gradients/huber_loss/Mul_3_grad/SumSum#gradients/huber_loss/Mul_3_grad/Mul5gradients/huber_loss/Mul_3_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
µ
'gradients/huber_loss/Mul_3_grad/ReshapeReshape#gradients/huber_loss/Mul_3_grad/Sum%gradients/huber_loss/Mul_3_grad/Shape*
T0*
Tshape0*
_output_shapes

: 
u
%gradients/huber_loss/Mul_3_grad/Mul_1Mulhuber_loss/Addgradients/AddN*
T0*
_output_shapes

: 
ћ
%gradients/huber_loss/Mul_3_grad/Sum_1Sum%gradients/huber_loss/Mul_3_grad/Mul_17gradients/huber_loss/Mul_3_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
≥
)gradients/huber_loss/Mul_3_grad/Reshape_1Reshape%gradients/huber_loss/Mul_3_grad/Sum_1'gradients/huber_loss/Mul_3_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
О
0gradients/huber_loss/Mul_3_grad/tuple/group_depsNoOp(^gradients/huber_loss/Mul_3_grad/Reshape*^gradients/huber_loss/Mul_3_grad/Reshape_1
Е
8gradients/huber_loss/Mul_3_grad/tuple/control_dependencyIdentity'gradients/huber_loss/Mul_3_grad/Reshape1^gradients/huber_loss/Mul_3_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/huber_loss/Mul_3_grad/Reshape*
_output_shapes

: 
Г
:gradients/huber_loss/Mul_3_grad/tuple/control_dependency_1Identity)gradients/huber_loss/Mul_3_grad/Reshape_11^gradients/huber_loss/Mul_3_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/huber_loss/Mul_3_grad/Reshape_1*
_output_shapes
: 
q
.gradients/huber_loss/Add_grad/tuple/group_depsNoOp9^gradients/huber_loss/Mul_3_grad/tuple/control_dependency
Т
6gradients/huber_loss/Add_grad/tuple/control_dependencyIdentity8gradients/huber_loss/Mul_3_grad/tuple/control_dependency/^gradients/huber_loss/Add_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/huber_loss/Mul_3_grad/Reshape*
_output_shapes

: 
Ф
8gradients/huber_loss/Add_grad/tuple/control_dependency_1Identity8gradients/huber_loss/Mul_3_grad/tuple/control_dependency/^gradients/huber_loss/Add_grad/tuple/group_deps*
_output_shapes

: *
T0*:
_class0
.,loc:@gradients/huber_loss/Mul_3_grad/Reshape
h
%gradients/huber_loss/Mul_1_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
x
'gradients/huber_loss/Mul_1_grad/Shape_1Const*
valueB"       *
dtype0*
_output_shapes
:
џ
5gradients/huber_loss/Mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs%gradients/huber_loss/Mul_1_grad/Shape'gradients/huber_loss/Mul_1_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Ы
#gradients/huber_loss/Mul_1_grad/MulMul6gradients/huber_loss/Add_grad/tuple/control_dependencyhuber_loss/Mul*
T0*
_output_shapes

: 
∆
#gradients/huber_loss/Mul_1_grad/SumSum#gradients/huber_loss/Mul_1_grad/Mul5gradients/huber_loss/Mul_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
≠
'gradients/huber_loss/Mul_1_grad/ReshapeReshape#gradients/huber_loss/Mul_1_grad/Sum%gradients/huber_loss/Mul_1_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
Я
%gradients/huber_loss/Mul_1_grad/Mul_1Mulhuber_loss/Const6gradients/huber_loss/Add_grad/tuple/control_dependency*
T0*
_output_shapes

: 
ћ
%gradients/huber_loss/Mul_1_grad/Sum_1Sum%gradients/huber_loss/Mul_1_grad/Mul_17gradients/huber_loss/Mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ї
)gradients/huber_loss/Mul_1_grad/Reshape_1Reshape%gradients/huber_loss/Mul_1_grad/Sum_1'gradients/huber_loss/Mul_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes

: 
О
0gradients/huber_loss/Mul_1_grad/tuple/group_depsNoOp(^gradients/huber_loss/Mul_1_grad/Reshape*^gradients/huber_loss/Mul_1_grad/Reshape_1
э
8gradients/huber_loss/Mul_1_grad/tuple/control_dependencyIdentity'gradients/huber_loss/Mul_1_grad/Reshape1^gradients/huber_loss/Mul_1_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/huber_loss/Mul_1_grad/Reshape*
_output_shapes
: 
Л
:gradients/huber_loss/Mul_1_grad/tuple/control_dependency_1Identity)gradients/huber_loss/Mul_1_grad/Reshape_11^gradients/huber_loss/Mul_1_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/huber_loss/Mul_1_grad/Reshape_1*
_output_shapes

: 
h
%gradients/huber_loss/Mul_2_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
x
'gradients/huber_loss/Mul_2_grad/Shape_1Const*
valueB"       *
dtype0*
_output_shapes
:
џ
5gradients/huber_loss/Mul_2_grad/BroadcastGradientArgsBroadcastGradientArgs%gradients/huber_loss/Mul_2_grad/Shape'gradients/huber_loss/Mul_2_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Я
#gradients/huber_loss/Mul_2_grad/MulMul8gradients/huber_loss/Add_grad/tuple/control_dependency_1huber_loss/Sub_1*
T0*
_output_shapes

: 
∆
#gradients/huber_loss/Mul_2_grad/SumSum#gradients/huber_loss/Mul_2_grad/Mul5gradients/huber_loss/Mul_2_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
≠
'gradients/huber_loss/Mul_2_grad/ReshapeReshape#gradients/huber_loss/Mul_2_grad/Sum%gradients/huber_loss/Mul_2_grad/Shape*
_output_shapes
: *
T0*
Tshape0
£
%gradients/huber_loss/Mul_2_grad/Mul_1Mulhuber_loss/Mul_2/x8gradients/huber_loss/Add_grad/tuple/control_dependency_1*
T0*
_output_shapes

: 
ћ
%gradients/huber_loss/Mul_2_grad/Sum_1Sum%gradients/huber_loss/Mul_2_grad/Mul_17gradients/huber_loss/Mul_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ї
)gradients/huber_loss/Mul_2_grad/Reshape_1Reshape%gradients/huber_loss/Mul_2_grad/Sum_1'gradients/huber_loss/Mul_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes

: 
О
0gradients/huber_loss/Mul_2_grad/tuple/group_depsNoOp(^gradients/huber_loss/Mul_2_grad/Reshape*^gradients/huber_loss/Mul_2_grad/Reshape_1
э
8gradients/huber_loss/Mul_2_grad/tuple/control_dependencyIdentity'gradients/huber_loss/Mul_2_grad/Reshape1^gradients/huber_loss/Mul_2_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/huber_loss/Mul_2_grad/Reshape*
_output_shapes
: 
Л
:gradients/huber_loss/Mul_2_grad/tuple/control_dependency_1Identity)gradients/huber_loss/Mul_2_grad/Reshape_11^gradients/huber_loss/Mul_2_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/huber_loss/Mul_2_grad/Reshape_1*
_output_shapes

: 
°
!gradients/huber_loss/Mul_grad/MulMul:gradients/huber_loss/Mul_1_grad/tuple/control_dependency_1huber_loss/Minimum*
T0*
_output_shapes

: 
£
#gradients/huber_loss/Mul_grad/Mul_1Mul:gradients/huber_loss/Mul_1_grad/tuple/control_dependency_1huber_loss/Minimum*
T0*
_output_shapes

: 
А
.gradients/huber_loss/Mul_grad/tuple/group_depsNoOp"^gradients/huber_loss/Mul_grad/Mul$^gradients/huber_loss/Mul_grad/Mul_1
х
6gradients/huber_loss/Mul_grad/tuple/control_dependencyIdentity!gradients/huber_loss/Mul_grad/Mul/^gradients/huber_loss/Mul_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients/huber_loss/Mul_grad/Mul*
_output_shapes

: 
ы
8gradients/huber_loss/Mul_grad/tuple/control_dependency_1Identity#gradients/huber_loss/Mul_grad/Mul_1/^gradients/huber_loss/Mul_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients/huber_loss/Mul_grad/Mul_1*
_output_shapes

: 
П
#gradients/huber_loss/Sub_1_grad/NegNeg:gradients/huber_loss/Mul_2_grad/tuple/control_dependency_1*
T0*
_output_shapes

: 
Ы
0gradients/huber_loss/Sub_1_grad/tuple/group_depsNoOp;^gradients/huber_loss/Mul_2_grad/tuple/control_dependency_1$^gradients/huber_loss/Sub_1_grad/Neg
Ъ
8gradients/huber_loss/Sub_1_grad/tuple/control_dependencyIdentity:gradients/huber_loss/Mul_2_grad/tuple/control_dependency_11^gradients/huber_loss/Sub_1_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/huber_loss/Mul_2_grad/Reshape_1*
_output_shapes

: 
€
:gradients/huber_loss/Sub_1_grad/tuple/control_dependency_1Identity#gradients/huber_loss/Sub_1_grad/Neg1^gradients/huber_loss/Sub_1_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients/huber_loss/Sub_1_grad/Neg*
_output_shapes

: 
Ѓ
gradients/AddN_1AddN6gradients/huber_loss/Mul_grad/tuple/control_dependency8gradients/huber_loss/Mul_grad/tuple/control_dependency_1:gradients/huber_loss/Sub_1_grad/tuple/control_dependency_1*
N*
_output_shapes

: *
T0*4
_class*
(&loc:@gradients/huber_loss/Mul_grad/Mul
x
'gradients/huber_loss/Minimum_grad/ShapeConst*
valueB"       *
dtype0*
_output_shapes
:
l
)gradients/huber_loss/Minimum_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
z
)gradients/huber_loss/Minimum_grad/Shape_2Const*
valueB"       *
dtype0*
_output_shapes
:
r
-gradients/huber_loss/Minimum_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ƒ
'gradients/huber_loss/Minimum_grad/zerosFill)gradients/huber_loss/Minimum_grad/Shape_2-gradients/huber_loss/Minimum_grad/zeros/Const*
T0*

index_type0*
_output_shapes

: 
З
+gradients/huber_loss/Minimum_grad/LessEqual	LessEqualhuber_loss/Abshuber_loss/Minimum/y*
T0*
_output_shapes

: 
б
7gradients/huber_loss/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgs'gradients/huber_loss/Minimum_grad/Shape)gradients/huber_loss/Minimum_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
√
(gradients/huber_loss/Minimum_grad/SelectSelect+gradients/huber_loss/Minimum_grad/LessEqualgradients/AddN_1'gradients/huber_loss/Minimum_grad/zeros*
T0*
_output_shapes

: 
≈
*gradients/huber_loss/Minimum_grad/Select_1Select+gradients/huber_loss/Minimum_grad/LessEqual'gradients/huber_loss/Minimum_grad/zerosgradients/AddN_1*
T0*
_output_shapes

: 
ѕ
%gradients/huber_loss/Minimum_grad/SumSum(gradients/huber_loss/Minimum_grad/Select7gradients/huber_loss/Minimum_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ї
)gradients/huber_loss/Minimum_grad/ReshapeReshape%gradients/huber_loss/Minimum_grad/Sum'gradients/huber_loss/Minimum_grad/Shape*
T0*
Tshape0*
_output_shapes

: 
’
'gradients/huber_loss/Minimum_grad/Sum_1Sum*gradients/huber_loss/Minimum_grad/Select_19gradients/huber_loss/Minimum_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
є
+gradients/huber_loss/Minimum_grad/Reshape_1Reshape'gradients/huber_loss/Minimum_grad/Sum_1)gradients/huber_loss/Minimum_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
Ф
2gradients/huber_loss/Minimum_grad/tuple/group_depsNoOp*^gradients/huber_loss/Minimum_grad/Reshape,^gradients/huber_loss/Minimum_grad/Reshape_1
Н
:gradients/huber_loss/Minimum_grad/tuple/control_dependencyIdentity)gradients/huber_loss/Minimum_grad/Reshape3^gradients/huber_loss/Minimum_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/huber_loss/Minimum_grad/Reshape*
_output_shapes

: 
Л
<gradients/huber_loss/Minimum_grad/tuple/control_dependency_1Identity+gradients/huber_loss/Minimum_grad/Reshape_13^gradients/huber_loss/Minimum_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/huber_loss/Minimum_grad/Reshape_1*
_output_shapes
: 
ю
gradients/AddN_2AddN8gradients/huber_loss/Sub_1_grad/tuple/control_dependency:gradients/huber_loss/Minimum_grad/tuple/control_dependency*
T0*<
_class2
0.loc:@gradients/huber_loss/Mul_2_grad/Reshape_1*
N*
_output_shapes

: 
c
"gradients/huber_loss/Abs_grad/SignSignhuber_loss/Sub*
T0*
_output_shapes

: 
З
!gradients/huber_loss/Abs_grad/mulMulgradients/AddN_2"gradients/huber_loss/Abs_grad/Sign*
T0*
_output_shapes

: 
f
#gradients/huber_loss/Sub_grad/ShapeShapeSum*
_output_shapes
:*
T0*
out_type0
r
%gradients/huber_loss/Sub_grad/Shape_1ShapePlaceholder_2*
T0*
out_type0*
_output_shapes
:
’
3gradients/huber_loss/Sub_grad/BroadcastGradientArgsBroadcastGradientArgs#gradients/huber_loss/Sub_grad/Shape%gradients/huber_loss/Sub_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
ј
!gradients/huber_loss/Sub_grad/SumSum!gradients/huber_loss/Abs_grad/mul3gradients/huber_loss/Sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Є
%gradients/huber_loss/Sub_grad/ReshapeReshape!gradients/huber_loss/Sub_grad/Sum#gradients/huber_loss/Sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
: €€€€€€€€€
ƒ
#gradients/huber_loss/Sub_grad/Sum_1Sum!gradients/huber_loss/Abs_grad/mul5gradients/huber_loss/Sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
p
!gradients/huber_loss/Sub_grad/NegNeg#gradients/huber_loss/Sub_grad/Sum_1*
_output_shapes
:*
T0
Љ
'gradients/huber_loss/Sub_grad/Reshape_1Reshape!gradients/huber_loss/Sub_grad/Neg%gradients/huber_loss/Sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
И
.gradients/huber_loss/Sub_grad/tuple/group_depsNoOp&^gradients/huber_loss/Sub_grad/Reshape(^gradients/huber_loss/Sub_grad/Reshape_1
Ж
6gradients/huber_loss/Sub_grad/tuple/control_dependencyIdentity%gradients/huber_loss/Sub_grad/Reshape/^gradients/huber_loss/Sub_grad/tuple/group_deps*
T0*8
_class.
,*loc:@gradients/huber_loss/Sub_grad/Reshape*'
_output_shapes
: €€€€€€€€€
М
8gradients/huber_loss/Sub_grad/tuple/control_dependency_1Identity'gradients/huber_loss/Sub_grad/Reshape_1/^gradients/huber_loss/Sub_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/huber_loss/Sub_grad/Reshape_1*'
_output_shapes
:€€€€€€€€€
[
gradients/Sum_grad/ShapeShapemul*
T0*
out_type0*
_output_shapes
:
Ж
gradients/Sum_grad/SizeConst*+
_class!
loc:@gradients/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
Ы
gradients/Sum_grad/addAddSum/reduction_indicesgradients/Sum_grad/Size*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
: 
°
gradients/Sum_grad/modFloorModgradients/Sum_grad/addgradients/Sum_grad/Size*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
: 
К
gradients/Sum_grad/Shape_1Const*
dtype0*
_output_shapes
: *+
_class!
loc:@gradients/Sum_grad/Shape*
valueB 
Н
gradients/Sum_grad/range/startConst*+
_class!
loc:@gradients/Sum_grad/Shape*
value	B : *
dtype0*
_output_shapes
: 
Н
gradients/Sum_grad/range/deltaConst*
dtype0*
_output_shapes
: *+
_class!
loc:@gradients/Sum_grad/Shape*
value	B :
ѕ
gradients/Sum_grad/rangeRangegradients/Sum_grad/range/startgradients/Sum_grad/Sizegradients/Sum_grad/range/delta*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
:*

Tidx0
М
gradients/Sum_grad/Fill/valueConst*+
_class!
loc:@gradients/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
Ї
gradients/Sum_grad/FillFillgradients/Sum_grad/Shape_1gradients/Sum_grad/Fill/value*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*

index_type0*
_output_shapes
: 
ъ
 gradients/Sum_grad/DynamicStitchDynamicStitchgradients/Sum_grad/rangegradients/Sum_grad/modgradients/Sum_grad/Shapegradients/Sum_grad/Fill*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*
N*#
_output_shapes
:€€€€€€€€€
Л
gradients/Sum_grad/Maximum/yConst*
dtype0*
_output_shapes
: *+
_class!
loc:@gradients/Sum_grad/Shape*
value	B :
ј
gradients/Sum_grad/MaximumMaximum gradients/Sum_grad/DynamicStitchgradients/Sum_grad/Maximum/y*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*#
_output_shapes
:€€€€€€€€€
ѓ
gradients/Sum_grad/floordivFloorDivgradients/Sum_grad/Shapegradients/Sum_grad/Maximum*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
:
∞
gradients/Sum_grad/ReshapeReshape6gradients/huber_loss/Sub_grad/tuple/control_dependency gradients/Sum_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
†
gradients/Sum_grad/TileTilegradients/Sum_grad/Reshapegradients/Sum_grad/floordiv*

Tmultiples0*
T0*+
_output_shapes
: €€€€€€€€€
f
gradients/mul_grad/ShapeShapemain/transpose*
_output_shapes
:*
T0*
out_type0
d
gradients/mul_grad/Shape_1Shape
ExpandDims*
T0*
out_type0*
_output_shapes
:
і
(gradients/mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_grad/Shapegradients/mul_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
x
gradients/mul_grad/MulMulgradients/Sum_grad/Tile
ExpandDims*
T0*+
_output_shapes
: €€€€€€€€€
Я
gradients/mul_grad/SumSumgradients/mul_grad/Mul(gradients/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ы
gradients/mul_grad/ReshapeReshapegradients/mul_grad/Sumgradients/mul_grad/Shape*
T0*
Tshape0*+
_output_shapes
: €€€€€€€€€
~
gradients/mul_grad/Mul_1Mulmain/transposegradients/Sum_grad/Tile*+
_output_shapes
: €€€€€€€€€*
T0
•
gradients/mul_grad/Sum_1Sumgradients/mul_grad/Mul_1*gradients/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
°
gradients/mul_grad/Reshape_1Reshapegradients/mul_grad/Sum_1gradients/mul_grad/Shape_1*+
_output_shapes
:€€€€€€€€€*
T0*
Tshape0
g
#gradients/mul_grad/tuple/group_depsNoOp^gradients/mul_grad/Reshape^gradients/mul_grad/Reshape_1
ё
+gradients/mul_grad/tuple/control_dependencyIdentitygradients/mul_grad/Reshape$^gradients/mul_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/mul_grad/Reshape*+
_output_shapes
: €€€€€€€€€
д
-gradients/mul_grad/tuple/control_dependency_1Identitygradients/mul_grad/Reshape_1$^gradients/mul_grad/tuple/group_deps*+
_output_shapes
:€€€€€€€€€*
T0*/
_class%
#!loc:@gradients/mul_grad/Reshape_1
~
/gradients/main/transpose_grad/InvertPermutationInvertPermutationmain/transpose/perm*
T0*
_output_shapes
:
’
'gradients/main/transpose_grad/transpose	Transpose+gradients/mul_grad/tuple/control_dependency/gradients/main/transpose_grad/InvertPermutation*
T0*+
_output_shapes
: €€€€€€€€€*
Tperm0
у
'gradients/main/transpose/x_grad/unstackUnpack'gradients/main/transpose_grad/transpose*ц
_output_shapesг
а:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€*	
num *
T0*

axis 
b
0gradients/main/transpose/x_grad/tuple/group_depsNoOp(^gradients/main/transpose/x_grad/unstack
О
8gradients/main/transpose/x_grad/tuple/control_dependencyIdentity'gradients/main/transpose/x_grad/unstack1^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:€€€€€€€€€
Т
:gradients/main/transpose/x_grad/tuple/control_dependency_1Identity)gradients/main/transpose/x_grad/unstack:11^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:€€€€€€€€€
Т
:gradients/main/transpose/x_grad/tuple/control_dependency_2Identity)gradients/main/transpose/x_grad/unstack:21^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:€€€€€€€€€
Т
:gradients/main/transpose/x_grad/tuple/control_dependency_3Identity)gradients/main/transpose/x_grad/unstack:31^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:€€€€€€€€€
Т
:gradients/main/transpose/x_grad/tuple/control_dependency_4Identity)gradients/main/transpose/x_grad/unstack:41^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:€€€€€€€€€
Т
:gradients/main/transpose/x_grad/tuple/control_dependency_5Identity)gradients/main/transpose/x_grad/unstack:51^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:€€€€€€€€€
Т
:gradients/main/transpose/x_grad/tuple/control_dependency_6Identity)gradients/main/transpose/x_grad/unstack:61^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:€€€€€€€€€
Т
:gradients/main/transpose/x_grad/tuple/control_dependency_7Identity)gradients/main/transpose/x_grad/unstack:71^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:€€€€€€€€€
Т
:gradients/main/transpose/x_grad/tuple/control_dependency_8Identity)gradients/main/transpose/x_grad/unstack:81^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:€€€€€€€€€
Т
:gradients/main/transpose/x_grad/tuple/control_dependency_9Identity)gradients/main/transpose/x_grad/unstack:91^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:€€€€€€€€€
Ф
;gradients/main/transpose/x_grad/tuple/control_dependency_10Identity*gradients/main/transpose/x_grad/unstack:101^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:€€€€€€€€€
Ф
;gradients/main/transpose/x_grad/tuple/control_dependency_11Identity*gradients/main/transpose/x_grad/unstack:111^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:€€€€€€€€€
Ф
;gradients/main/transpose/x_grad/tuple/control_dependency_12Identity*gradients/main/transpose/x_grad/unstack:121^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:€€€€€€€€€
Ф
;gradients/main/transpose/x_grad/tuple/control_dependency_13Identity*gradients/main/transpose/x_grad/unstack:131^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:€€€€€€€€€
Ф
;gradients/main/transpose/x_grad/tuple/control_dependency_14Identity*gradients/main/transpose/x_grad/unstack:141^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:€€€€€€€€€
Ф
;gradients/main/transpose/x_grad/tuple/control_dependency_15Identity*gradients/main/transpose/x_grad/unstack:151^gradients/main/transpose/x_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack
Ф
;gradients/main/transpose/x_grad/tuple/control_dependency_16Identity*gradients/main/transpose/x_grad/unstack:161^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:€€€€€€€€€
Ф
;gradients/main/transpose/x_grad/tuple/control_dependency_17Identity*gradients/main/transpose/x_grad/unstack:171^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:€€€€€€€€€
Ф
;gradients/main/transpose/x_grad/tuple/control_dependency_18Identity*gradients/main/transpose/x_grad/unstack:181^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:€€€€€€€€€
Ф
;gradients/main/transpose/x_grad/tuple/control_dependency_19Identity*gradients/main/transpose/x_grad/unstack:191^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:€€€€€€€€€
Ф
;gradients/main/transpose/x_grad/tuple/control_dependency_20Identity*gradients/main/transpose/x_grad/unstack:201^gradients/main/transpose/x_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack
Ф
;gradients/main/transpose/x_grad/tuple/control_dependency_21Identity*gradients/main/transpose/x_grad/unstack:211^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:€€€€€€€€€
Ф
;gradients/main/transpose/x_grad/tuple/control_dependency_22Identity*gradients/main/transpose/x_grad/unstack:221^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:€€€€€€€€€
Ф
;gradients/main/transpose/x_grad/tuple/control_dependency_23Identity*gradients/main/transpose/x_grad/unstack:231^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:€€€€€€€€€
Ф
;gradients/main/transpose/x_grad/tuple/control_dependency_24Identity*gradients/main/transpose/x_grad/unstack:241^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:€€€€€€€€€
Ф
;gradients/main/transpose/x_grad/tuple/control_dependency_25Identity*gradients/main/transpose/x_grad/unstack:251^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:€€€€€€€€€
Ф
;gradients/main/transpose/x_grad/tuple/control_dependency_26Identity*gradients/main/transpose/x_grad/unstack:261^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:€€€€€€€€€
Ф
;gradients/main/transpose/x_grad/tuple/control_dependency_27Identity*gradients/main/transpose/x_grad/unstack:271^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:€€€€€€€€€
Ф
;gradients/main/transpose/x_grad/tuple/control_dependency_28Identity*gradients/main/transpose/x_grad/unstack:281^gradients/main/transpose/x_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack
Ф
;gradients/main/transpose/x_grad/tuple/control_dependency_29Identity*gradients/main/transpose/x_grad/unstack:291^gradients/main/transpose/x_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack
Ф
;gradients/main/transpose/x_grad/tuple/control_dependency_30Identity*gradients/main/transpose/x_grad/unstack:301^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:€€€€€€€€€
Ф
;gradients/main/transpose/x_grad/tuple/control_dependency_31Identity*gradients/main/transpose/x_grad/unstack:311^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:€€€€€€€€€
Э
 gradients/main/split_grad/concatConcatV28gradients/main/transpose/x_grad/tuple/control_dependency:gradients/main/transpose/x_grad/tuple/control_dependency_1:gradients/main/transpose/x_grad/tuple/control_dependency_2:gradients/main/transpose/x_grad/tuple/control_dependency_3:gradients/main/transpose/x_grad/tuple/control_dependency_4:gradients/main/transpose/x_grad/tuple/control_dependency_5:gradients/main/transpose/x_grad/tuple/control_dependency_6:gradients/main/transpose/x_grad/tuple/control_dependency_7:gradients/main/transpose/x_grad/tuple/control_dependency_8:gradients/main/transpose/x_grad/tuple/control_dependency_9;gradients/main/transpose/x_grad/tuple/control_dependency_10;gradients/main/transpose/x_grad/tuple/control_dependency_11;gradients/main/transpose/x_grad/tuple/control_dependency_12;gradients/main/transpose/x_grad/tuple/control_dependency_13;gradients/main/transpose/x_grad/tuple/control_dependency_14;gradients/main/transpose/x_grad/tuple/control_dependency_15;gradients/main/transpose/x_grad/tuple/control_dependency_16;gradients/main/transpose/x_grad/tuple/control_dependency_17;gradients/main/transpose/x_grad/tuple/control_dependency_18;gradients/main/transpose/x_grad/tuple/control_dependency_19;gradients/main/transpose/x_grad/tuple/control_dependency_20;gradients/main/transpose/x_grad/tuple/control_dependency_21;gradients/main/transpose/x_grad/tuple/control_dependency_22;gradients/main/transpose/x_grad/tuple/control_dependency_23;gradients/main/transpose/x_grad/tuple/control_dependency_24;gradients/main/transpose/x_grad/tuple/control_dependency_25;gradients/main/transpose/x_grad/tuple/control_dependency_26;gradients/main/transpose/x_grad/tuple/control_dependency_27;gradients/main/transpose/x_grad/tuple/control_dependency_28;gradients/main/transpose/x_grad/tuple/control_dependency_29;gradients/main/transpose/x_grad/tuple/control_dependency_30;gradients/main/transpose/x_grad/tuple/control_dependency_31main/split/split_dim*
T0*
N *'
_output_shapes
:€€€€€€€€€*

Tidx0
Ь
/gradients/main/dense_4/BiasAdd_grad/BiasAddGradBiasAddGrad gradients/main/split_grad/concat*
T0*
data_formatNHWC*
_output_shapes
:
С
4gradients/main/dense_4/BiasAdd_grad/tuple/group_depsNoOp0^gradients/main/dense_4/BiasAdd_grad/BiasAddGrad!^gradients/main/split_grad/concat
И
<gradients/main/dense_4/BiasAdd_grad/tuple/control_dependencyIdentity gradients/main/split_grad/concat5^gradients/main/dense_4/BiasAdd_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/main/split_grad/concat*'
_output_shapes
:€€€€€€€€€
Ы
>gradients/main/dense_4/BiasAdd_grad/tuple/control_dependency_1Identity/gradients/main/dense_4/BiasAdd_grad/BiasAddGrad5^gradients/main/dense_4/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*B
_class8
64loc:@gradients/main/dense_4/BiasAdd_grad/BiasAddGrad
д
)gradients/main/dense_4/MatMul_grad/MatMulMatMul<gradients/main/dense_4/BiasAdd_grad/tuple/control_dependencymain/dense_4/kernel/read*
T0*(
_output_shapes
:€€€€€€€€€А*
transpose_a( *
transpose_b(
÷
+gradients/main/dense_4/MatMul_grad/MatMul_1MatMulmain/dense_3/Relu<gradients/main/dense_4/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	А*
transpose_a(*
transpose_b( *
T0
Х
3gradients/main/dense_4/MatMul_grad/tuple/group_depsNoOp*^gradients/main/dense_4/MatMul_grad/MatMul,^gradients/main/dense_4/MatMul_grad/MatMul_1
Щ
;gradients/main/dense_4/MatMul_grad/tuple/control_dependencyIdentity)gradients/main/dense_4/MatMul_grad/MatMul4^gradients/main/dense_4/MatMul_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/main/dense_4/MatMul_grad/MatMul*(
_output_shapes
:€€€€€€€€€А
Ц
=gradients/main/dense_4/MatMul_grad/tuple/control_dependency_1Identity+gradients/main/dense_4/MatMul_grad/MatMul_14^gradients/main/dense_4/MatMul_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/main/dense_4/MatMul_grad/MatMul_1*
_output_shapes
:	А
Є
)gradients/main/dense_3/Relu_grad/ReluGradReluGrad;gradients/main/dense_4/MatMul_grad/tuple/control_dependencymain/dense_3/Relu*
T0*(
_output_shapes
:€€€€€€€€€А
¶
/gradients/main/dense_3/BiasAdd_grad/BiasAddGradBiasAddGrad)gradients/main/dense_3/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:А
Ъ
4gradients/main/dense_3/BiasAdd_grad/tuple/group_depsNoOp0^gradients/main/dense_3/BiasAdd_grad/BiasAddGrad*^gradients/main/dense_3/Relu_grad/ReluGrad
Ы
<gradients/main/dense_3/BiasAdd_grad/tuple/control_dependencyIdentity)gradients/main/dense_3/Relu_grad/ReluGrad5^gradients/main/dense_3/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:€€€€€€€€€А*
T0*<
_class2
0.loc:@gradients/main/dense_3/Relu_grad/ReluGrad
Ь
>gradients/main/dense_3/BiasAdd_grad/tuple/control_dependency_1Identity/gradients/main/dense_3/BiasAdd_grad/BiasAddGrad5^gradients/main/dense_3/BiasAdd_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/main/dense_3/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:А
д
)gradients/main/dense_3/MatMul_grad/MatMulMatMul<gradients/main/dense_3/BiasAdd_grad/tuple/control_dependencymain/dense_3/kernel/read*
transpose_b(*
T0*(
_output_shapes
:€€€€€€€€€А*
transpose_a( 
„
+gradients/main/dense_3/MatMul_grad/MatMul_1MatMulmain/dense_2/Relu<gradients/main/dense_3/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
АА*
transpose_a(*
transpose_b( *
T0
Х
3gradients/main/dense_3/MatMul_grad/tuple/group_depsNoOp*^gradients/main/dense_3/MatMul_grad/MatMul,^gradients/main/dense_3/MatMul_grad/MatMul_1
Щ
;gradients/main/dense_3/MatMul_grad/tuple/control_dependencyIdentity)gradients/main/dense_3/MatMul_grad/MatMul4^gradients/main/dense_3/MatMul_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/main/dense_3/MatMul_grad/MatMul*(
_output_shapes
:€€€€€€€€€А
Ч
=gradients/main/dense_3/MatMul_grad/tuple/control_dependency_1Identity+gradients/main/dense_3/MatMul_grad/MatMul_14^gradients/main/dense_3/MatMul_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/main/dense_3/MatMul_grad/MatMul_1* 
_output_shapes
:
АА
Є
)gradients/main/dense_2/Relu_grad/ReluGradReluGrad;gradients/main/dense_3/MatMul_grad/tuple/control_dependencymain/dense_2/Relu*
T0*(
_output_shapes
:€€€€€€€€€А
¶
/gradients/main/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad)gradients/main/dense_2/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:А
Ъ
4gradients/main/dense_2/BiasAdd_grad/tuple/group_depsNoOp0^gradients/main/dense_2/BiasAdd_grad/BiasAddGrad*^gradients/main/dense_2/Relu_grad/ReluGrad
Ы
<gradients/main/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity)gradients/main/dense_2/Relu_grad/ReluGrad5^gradients/main/dense_2/BiasAdd_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/main/dense_2/Relu_grad/ReluGrad*(
_output_shapes
:€€€€€€€€€А
Ь
>gradients/main/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity/gradients/main/dense_2/BiasAdd_grad/BiasAddGrad5^gradients/main/dense_2/BiasAdd_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/main/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:А
д
)gradients/main/dense_2/MatMul_grad/MatMulMatMul<gradients/main/dense_2/BiasAdd_grad/tuple/control_dependencymain/dense_2/kernel/read*
T0*(
_output_shapes
:€€€€€€€€€А*
transpose_a( *
transpose_b(
ќ
+gradients/main/dense_2/MatMul_grad/MatMul_1MatMulmain/Mul<gradients/main/dense_2/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
АА*
transpose_a(*
transpose_b( 
Х
3gradients/main/dense_2/MatMul_grad/tuple/group_depsNoOp*^gradients/main/dense_2/MatMul_grad/MatMul,^gradients/main/dense_2/MatMul_grad/MatMul_1
Щ
;gradients/main/dense_2/MatMul_grad/tuple/control_dependencyIdentity)gradients/main/dense_2/MatMul_grad/MatMul4^gradients/main/dense_2/MatMul_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/main/dense_2/MatMul_grad/MatMul*(
_output_shapes
:€€€€€€€€€А
Ч
=gradients/main/dense_2/MatMul_grad/tuple/control_dependency_1Identity+gradients/main/dense_2/MatMul_grad/MatMul_14^gradients/main/dense_2/MatMul_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/main/dense_2/MatMul_grad/MatMul_1* 
_output_shapes
:
АА
l
gradients/main/Mul_grad/ShapeShapemain/dense/Selu*
T0*
out_type0*
_output_shapes
:
p
gradients/main/Mul_grad/Shape_1Shapemain/dense_1/Relu*
T0*
out_type0*
_output_shapes
:
√
-gradients/main/Mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/main/Mul_grad/Shapegradients/main/Mul_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
•
gradients/main/Mul_grad/MulMul;gradients/main/dense_2/MatMul_grad/tuple/control_dependencymain/dense_1/Relu*(
_output_shapes
:€€€€€€€€€А*
T0
Ѓ
gradients/main/Mul_grad/SumSumgradients/main/Mul_grad/Mul-gradients/main/Mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
І
gradients/main/Mul_grad/ReshapeReshapegradients/main/Mul_grad/Sumgradients/main/Mul_grad/Shape*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
•
gradients/main/Mul_grad/Mul_1Mulmain/dense/Selu;gradients/main/dense_2/MatMul_grad/tuple/control_dependency*(
_output_shapes
:€€€€€€€€€А*
T0
і
gradients/main/Mul_grad/Sum_1Sumgradients/main/Mul_grad/Mul_1/gradients/main/Mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
≠
!gradients/main/Mul_grad/Reshape_1Reshapegradients/main/Mul_grad/Sum_1gradients/main/Mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
v
(gradients/main/Mul_grad/tuple/group_depsNoOp ^gradients/main/Mul_grad/Reshape"^gradients/main/Mul_grad/Reshape_1
п
0gradients/main/Mul_grad/tuple/control_dependencyIdentitygradients/main/Mul_grad/Reshape)^gradients/main/Mul_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/main/Mul_grad/Reshape*(
_output_shapes
:€€€€€€€€€А
х
2gradients/main/Mul_grad/tuple/control_dependency_1Identity!gradients/main/Mul_grad/Reshape_1)^gradients/main/Mul_grad/tuple/group_deps*(
_output_shapes
:€€€€€€€€€А*
T0*4
_class*
(&loc:@gradients/main/Mul_grad/Reshape_1
©
'gradients/main/dense/Selu_grad/SeluGradSeluGrad0gradients/main/Mul_grad/tuple/control_dependencymain/dense/Selu*
T0*(
_output_shapes
:€€€€€€€€€А
ѓ
)gradients/main/dense_1/Relu_grad/ReluGradReluGrad2gradients/main/Mul_grad/tuple/control_dependency_1main/dense_1/Relu*
T0*(
_output_shapes
:€€€€€€€€€А
Ґ
-gradients/main/dense/BiasAdd_grad/BiasAddGradBiasAddGrad'gradients/main/dense/Selu_grad/SeluGrad*
data_formatNHWC*
_output_shapes	
:А*
T0
Ф
2gradients/main/dense/BiasAdd_grad/tuple/group_depsNoOp.^gradients/main/dense/BiasAdd_grad/BiasAddGrad(^gradients/main/dense/Selu_grad/SeluGrad
У
:gradients/main/dense/BiasAdd_grad/tuple/control_dependencyIdentity'gradients/main/dense/Selu_grad/SeluGrad3^gradients/main/dense/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:€€€€€€€€€А*
T0*:
_class0
.,loc:@gradients/main/dense/Selu_grad/SeluGrad
Ф
<gradients/main/dense/BiasAdd_grad/tuple/control_dependency_1Identity-gradients/main/dense/BiasAdd_grad/BiasAddGrad3^gradients/main/dense/BiasAdd_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/main/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:А
¶
/gradients/main/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad)gradients/main/dense_1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:А
Ъ
4gradients/main/dense_1/BiasAdd_grad/tuple/group_depsNoOp0^gradients/main/dense_1/BiasAdd_grad/BiasAddGrad*^gradients/main/dense_1/Relu_grad/ReluGrad
Ы
<gradients/main/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity)gradients/main/dense_1/Relu_grad/ReluGrad5^gradients/main/dense_1/BiasAdd_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/main/dense_1/Relu_grad/ReluGrad*(
_output_shapes
:€€€€€€€€€А
Ь
>gradients/main/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity/gradients/main/dense_1/BiasAdd_grad/BiasAddGrad5^gradients/main/dense_1/BiasAdd_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/main/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:А
Ё
'gradients/main/dense/MatMul_grad/MatMulMatMul:gradients/main/dense/BiasAdd_grad/tuple/control_dependencymain/dense/kernel/read*'
_output_shapes
:€€€€€€€€€*
transpose_a( *
transpose_b(*
T0
Ќ
)gradients/main/dense/MatMul_grad/MatMul_1MatMulmain/Reshape:gradients/main/dense/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes
:	А*
transpose_a(
П
1gradients/main/dense/MatMul_grad/tuple/group_depsNoOp(^gradients/main/dense/MatMul_grad/MatMul*^gradients/main/dense/MatMul_grad/MatMul_1
Р
9gradients/main/dense/MatMul_grad/tuple/control_dependencyIdentity'gradients/main/dense/MatMul_grad/MatMul2^gradients/main/dense/MatMul_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/dense/MatMul_grad/MatMul*'
_output_shapes
:€€€€€€€€€
О
;gradients/main/dense/MatMul_grad/tuple/control_dependency_1Identity)gradients/main/dense/MatMul_grad/MatMul_12^gradients/main/dense/MatMul_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/main/dense/MatMul_grad/MatMul_1*
_output_shapes
:	А
г
)gradients/main/dense_1/MatMul_grad/MatMulMatMul<gradients/main/dense_1/BiasAdd_grad/tuple/control_dependencymain/dense_1/kernel/read*
T0*'
_output_shapes
:€€€€€€€€€@*
transpose_a( *
transpose_b(
Ќ
+gradients/main/dense_1/MatMul_grad/MatMul_1MatMulmain/Cos<gradients/main/dense_1/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	@А*
transpose_a(*
transpose_b( *
T0
Х
3gradients/main/dense_1/MatMul_grad/tuple/group_depsNoOp*^gradients/main/dense_1/MatMul_grad/MatMul,^gradients/main/dense_1/MatMul_grad/MatMul_1
Ш
;gradients/main/dense_1/MatMul_grad/tuple/control_dependencyIdentity)gradients/main/dense_1/MatMul_grad/MatMul4^gradients/main/dense_1/MatMul_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/main/dense_1/MatMul_grad/MatMul*'
_output_shapes
:€€€€€€€€€@
Ц
=gradients/main/dense_1/MatMul_grad/tuple/control_dependency_1Identity+gradients/main/dense_1/MatMul_grad/MatMul_14^gradients/main/dense_1/MatMul_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/main/dense_1/MatMul_grad/MatMul_1*
_output_shapes
:	@А
В
beta1_power/initial_valueConst*"
_class
loc:@main/dense/bias*
valueB
 *fff?*
dtype0*
_output_shapes
: 
У
beta1_power
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *"
_class
loc:@main/dense/bias
≤
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*"
_class
loc:@main/dense/bias
n
beta1_power/readIdentitybeta1_power*
_output_shapes
: *
T0*"
_class
loc:@main/dense/bias
В
beta2_power/initial_valueConst*
dtype0*
_output_shapes
: *"
_class
loc:@main/dense/bias*
valueB
 *wЊ?
У
beta2_power
VariableV2*
shared_name *"
_class
loc:@main/dense/bias*
	container *
shape: *
dtype0*
_output_shapes
: 
≤
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
use_locking(*
T0*"
_class
loc:@main/dense/bias*
validate_shape(*
_output_shapes
: 
n
beta2_power/readIdentitybeta2_power*
T0*"
_class
loc:@main/dense/bias*
_output_shapes
: 
•
(main/dense/kernel/Adam/Initializer/zerosConst*$
_class
loc:@main/dense/kernel*
valueB	А*    *
dtype0*
_output_shapes
:	А
≤
main/dense/kernel/Adam
VariableV2*
	container *
shape:	А*
dtype0*
_output_shapes
:	А*
shared_name *$
_class
loc:@main/dense/kernel
в
main/dense/kernel/Adam/AssignAssignmain/dense/kernel/Adam(main/dense/kernel/Adam/Initializer/zeros*
T0*$
_class
loc:@main/dense/kernel*
validate_shape(*
_output_shapes
:	А*
use_locking(
П
main/dense/kernel/Adam/readIdentitymain/dense/kernel/Adam*
_output_shapes
:	А*
T0*$
_class
loc:@main/dense/kernel
І
*main/dense/kernel/Adam_1/Initializer/zerosConst*$
_class
loc:@main/dense/kernel*
valueB	А*    *
dtype0*
_output_shapes
:	А
і
main/dense/kernel/Adam_1
VariableV2*
	container *
shape:	А*
dtype0*
_output_shapes
:	А*
shared_name *$
_class
loc:@main/dense/kernel
и
main/dense/kernel/Adam_1/AssignAssignmain/dense/kernel/Adam_1*main/dense/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@main/dense/kernel*
validate_shape(*
_output_shapes
:	А
У
main/dense/kernel/Adam_1/readIdentitymain/dense/kernel/Adam_1*
T0*$
_class
loc:@main/dense/kernel*
_output_shapes
:	А
Щ
&main/dense/bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes	
:А*"
_class
loc:@main/dense/bias*
valueBА*    
¶
main/dense/bias/Adam
VariableV2*"
_class
loc:@main/dense/bias*
	container *
shape:А*
dtype0*
_output_shapes	
:А*
shared_name 
÷
main/dense/bias/Adam/AssignAssignmain/dense/bias/Adam&main/dense/bias/Adam/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@main/dense/bias*
validate_shape(*
_output_shapes	
:А
Е
main/dense/bias/Adam/readIdentitymain/dense/bias/Adam*
T0*"
_class
loc:@main/dense/bias*
_output_shapes	
:А
Ы
(main/dense/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes	
:А*"
_class
loc:@main/dense/bias*
valueBА*    
®
main/dense/bias/Adam_1
VariableV2*
shared_name *"
_class
loc:@main/dense/bias*
	container *
shape:А*
dtype0*
_output_shapes	
:А
№
main/dense/bias/Adam_1/AssignAssignmain/dense/bias/Adam_1(main/dense/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@main/dense/bias*
validate_shape(*
_output_shapes	
:А
Й
main/dense/bias/Adam_1/readIdentitymain/dense/bias/Adam_1*
T0*"
_class
loc:@main/dense/bias*
_output_shapes	
:А
≥
:main/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*&
_class
loc:@main/dense_1/kernel*
valueB"@   А   *
dtype0*
_output_shapes
:
Э
0main/dense_1/kernel/Adam/Initializer/zeros/ConstConst*&
_class
loc:@main/dense_1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Д
*main/dense_1/kernel/Adam/Initializer/zerosFill:main/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensor0main/dense_1/kernel/Adam/Initializer/zeros/Const*
T0*&
_class
loc:@main/dense_1/kernel*

index_type0*
_output_shapes
:	@А
ґ
main/dense_1/kernel/Adam
VariableV2*&
_class
loc:@main/dense_1/kernel*
	container *
shape:	@А*
dtype0*
_output_shapes
:	@А*
shared_name 
к
main/dense_1/kernel/Adam/AssignAssignmain/dense_1/kernel/Adam*main/dense_1/kernel/Adam/Initializer/zeros*
use_locking(*
T0*&
_class
loc:@main/dense_1/kernel*
validate_shape(*
_output_shapes
:	@А
Х
main/dense_1/kernel/Adam/readIdentitymain/dense_1/kernel/Adam*
_output_shapes
:	@А*
T0*&
_class
loc:@main/dense_1/kernel
µ
<main/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*&
_class
loc:@main/dense_1/kernel*
valueB"@   А   *
dtype0*
_output_shapes
:
Я
2main/dense_1/kernel/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *&
_class
loc:@main/dense_1/kernel*
valueB
 *    
К
,main/dense_1/kernel/Adam_1/Initializer/zerosFill<main/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor2main/dense_1/kernel/Adam_1/Initializer/zeros/Const*
T0*&
_class
loc:@main/dense_1/kernel*

index_type0*
_output_shapes
:	@А
Є
main/dense_1/kernel/Adam_1
VariableV2*
shared_name *&
_class
loc:@main/dense_1/kernel*
	container *
shape:	@А*
dtype0*
_output_shapes
:	@А
р
!main/dense_1/kernel/Adam_1/AssignAssignmain/dense_1/kernel/Adam_1,main/dense_1/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*&
_class
loc:@main/dense_1/kernel*
validate_shape(*
_output_shapes
:	@А
Щ
main/dense_1/kernel/Adam_1/readIdentitymain/dense_1/kernel/Adam_1*
T0*&
_class
loc:@main/dense_1/kernel*
_output_shapes
:	@А
Э
(main/dense_1/bias/Adam/Initializer/zerosConst*$
_class
loc:@main/dense_1/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
™
main/dense_1/bias/Adam
VariableV2*
dtype0*
_output_shapes	
:А*
shared_name *$
_class
loc:@main/dense_1/bias*
	container *
shape:А
ё
main/dense_1/bias/Adam/AssignAssignmain/dense_1/bias/Adam(main/dense_1/bias/Adam/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@main/dense_1/bias*
validate_shape(*
_output_shapes	
:А
Л
main/dense_1/bias/Adam/readIdentitymain/dense_1/bias/Adam*
T0*$
_class
loc:@main/dense_1/bias*
_output_shapes	
:А
Я
*main/dense_1/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes	
:А*$
_class
loc:@main/dense_1/bias*
valueBА*    
ђ
main/dense_1/bias/Adam_1
VariableV2*
dtype0*
_output_shapes	
:А*
shared_name *$
_class
loc:@main/dense_1/bias*
	container *
shape:А
д
main/dense_1/bias/Adam_1/AssignAssignmain/dense_1/bias/Adam_1*main/dense_1/bias/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0*$
_class
loc:@main/dense_1/bias
П
main/dense_1/bias/Adam_1/readIdentitymain/dense_1/bias/Adam_1*
_output_shapes	
:А*
T0*$
_class
loc:@main/dense_1/bias
≥
:main/dense_2/kernel/Adam/Initializer/zeros/shape_as_tensorConst*&
_class
loc:@main/dense_2/kernel*
valueB"А      *
dtype0*
_output_shapes
:
Э
0main/dense_2/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *&
_class
loc:@main/dense_2/kernel*
valueB
 *    
Е
*main/dense_2/kernel/Adam/Initializer/zerosFill:main/dense_2/kernel/Adam/Initializer/zeros/shape_as_tensor0main/dense_2/kernel/Adam/Initializer/zeros/Const*
T0*&
_class
loc:@main/dense_2/kernel*

index_type0* 
_output_shapes
:
АА
Є
main/dense_2/kernel/Adam
VariableV2*
dtype0* 
_output_shapes
:
АА*
shared_name *&
_class
loc:@main/dense_2/kernel*
	container *
shape:
АА
л
main/dense_2/kernel/Adam/AssignAssignmain/dense_2/kernel/Adam*main/dense_2/kernel/Adam/Initializer/zeros*
T0*&
_class
loc:@main/dense_2/kernel*
validate_shape(* 
_output_shapes
:
АА*
use_locking(
Ц
main/dense_2/kernel/Adam/readIdentitymain/dense_2/kernel/Adam*
T0*&
_class
loc:@main/dense_2/kernel* 
_output_shapes
:
АА
µ
<main/dense_2/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*&
_class
loc:@main/dense_2/kernel*
valueB"А      *
dtype0*
_output_shapes
:
Я
2main/dense_2/kernel/Adam_1/Initializer/zeros/ConstConst*&
_class
loc:@main/dense_2/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Л
,main/dense_2/kernel/Adam_1/Initializer/zerosFill<main/dense_2/kernel/Adam_1/Initializer/zeros/shape_as_tensor2main/dense_2/kernel/Adam_1/Initializer/zeros/Const* 
_output_shapes
:
АА*
T0*&
_class
loc:@main/dense_2/kernel*

index_type0
Ї
main/dense_2/kernel/Adam_1
VariableV2*
shared_name *&
_class
loc:@main/dense_2/kernel*
	container *
shape:
АА*
dtype0* 
_output_shapes
:
АА
с
!main/dense_2/kernel/Adam_1/AssignAssignmain/dense_2/kernel/Adam_1,main/dense_2/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*&
_class
loc:@main/dense_2/kernel*
validate_shape(* 
_output_shapes
:
АА
Ъ
main/dense_2/kernel/Adam_1/readIdentitymain/dense_2/kernel/Adam_1*
T0*&
_class
loc:@main/dense_2/kernel* 
_output_shapes
:
АА
Э
(main/dense_2/bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes	
:А*$
_class
loc:@main/dense_2/bias*
valueBА*    
™
main/dense_2/bias/Adam
VariableV2*
shape:А*
dtype0*
_output_shapes	
:А*
shared_name *$
_class
loc:@main/dense_2/bias*
	container 
ё
main/dense_2/bias/Adam/AssignAssignmain/dense_2/bias/Adam(main/dense_2/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0*$
_class
loc:@main/dense_2/bias
Л
main/dense_2/bias/Adam/readIdentitymain/dense_2/bias/Adam*
T0*$
_class
loc:@main/dense_2/bias*
_output_shapes	
:А
Я
*main/dense_2/bias/Adam_1/Initializer/zerosConst*$
_class
loc:@main/dense_2/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
ђ
main/dense_2/bias/Adam_1
VariableV2*
	container *
shape:А*
dtype0*
_output_shapes	
:А*
shared_name *$
_class
loc:@main/dense_2/bias
д
main/dense_2/bias/Adam_1/AssignAssignmain/dense_2/bias/Adam_1*main/dense_2/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@main/dense_2/bias*
validate_shape(*
_output_shapes	
:А
П
main/dense_2/bias/Adam_1/readIdentitymain/dense_2/bias/Adam_1*
T0*$
_class
loc:@main/dense_2/bias*
_output_shapes	
:А
≥
:main/dense_3/kernel/Adam/Initializer/zeros/shape_as_tensorConst*&
_class
loc:@main/dense_3/kernel*
valueB"   А   *
dtype0*
_output_shapes
:
Э
0main/dense_3/kernel/Adam/Initializer/zeros/ConstConst*&
_class
loc:@main/dense_3/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Е
*main/dense_3/kernel/Adam/Initializer/zerosFill:main/dense_3/kernel/Adam/Initializer/zeros/shape_as_tensor0main/dense_3/kernel/Adam/Initializer/zeros/Const*
T0*&
_class
loc:@main/dense_3/kernel*

index_type0* 
_output_shapes
:
АА
Є
main/dense_3/kernel/Adam
VariableV2*
dtype0* 
_output_shapes
:
АА*
shared_name *&
_class
loc:@main/dense_3/kernel*
	container *
shape:
АА
л
main/dense_3/kernel/Adam/AssignAssignmain/dense_3/kernel/Adam*main/dense_3/kernel/Adam/Initializer/zeros*
use_locking(*
T0*&
_class
loc:@main/dense_3/kernel*
validate_shape(* 
_output_shapes
:
АА
Ц
main/dense_3/kernel/Adam/readIdentitymain/dense_3/kernel/Adam*
T0*&
_class
loc:@main/dense_3/kernel* 
_output_shapes
:
АА
µ
<main/dense_3/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*&
_class
loc:@main/dense_3/kernel*
valueB"   А   
Я
2main/dense_3/kernel/Adam_1/Initializer/zeros/ConstConst*&
_class
loc:@main/dense_3/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Л
,main/dense_3/kernel/Adam_1/Initializer/zerosFill<main/dense_3/kernel/Adam_1/Initializer/zeros/shape_as_tensor2main/dense_3/kernel/Adam_1/Initializer/zeros/Const*
T0*&
_class
loc:@main/dense_3/kernel*

index_type0* 
_output_shapes
:
АА
Ї
main/dense_3/kernel/Adam_1
VariableV2*
	container *
shape:
АА*
dtype0* 
_output_shapes
:
АА*
shared_name *&
_class
loc:@main/dense_3/kernel
с
!main/dense_3/kernel/Adam_1/AssignAssignmain/dense_3/kernel/Adam_1,main/dense_3/kernel/Adam_1/Initializer/zeros*
validate_shape(* 
_output_shapes
:
АА*
use_locking(*
T0*&
_class
loc:@main/dense_3/kernel
Ъ
main/dense_3/kernel/Adam_1/readIdentitymain/dense_3/kernel/Adam_1*
T0*&
_class
loc:@main/dense_3/kernel* 
_output_shapes
:
АА
Э
(main/dense_3/bias/Adam/Initializer/zerosConst*$
_class
loc:@main/dense_3/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
™
main/dense_3/bias/Adam
VariableV2*$
_class
loc:@main/dense_3/bias*
	container *
shape:А*
dtype0*
_output_shapes	
:А*
shared_name 
ё
main/dense_3/bias/Adam/AssignAssignmain/dense_3/bias/Adam(main/dense_3/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0*$
_class
loc:@main/dense_3/bias
Л
main/dense_3/bias/Adam/readIdentitymain/dense_3/bias/Adam*
_output_shapes	
:А*
T0*$
_class
loc:@main/dense_3/bias
Я
*main/dense_3/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes	
:А*$
_class
loc:@main/dense_3/bias*
valueBА*    
ђ
main/dense_3/bias/Adam_1
VariableV2*
dtype0*
_output_shapes	
:А*
shared_name *$
_class
loc:@main/dense_3/bias*
	container *
shape:А
д
main/dense_3/bias/Adam_1/AssignAssignmain/dense_3/bias/Adam_1*main/dense_3/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@main/dense_3/bias*
validate_shape(*
_output_shapes	
:А
П
main/dense_3/bias/Adam_1/readIdentitymain/dense_3/bias/Adam_1*
_output_shapes	
:А*
T0*$
_class
loc:@main/dense_3/bias
©
*main/dense_4/kernel/Adam/Initializer/zerosConst*&
_class
loc:@main/dense_4/kernel*
valueB	А*    *
dtype0*
_output_shapes
:	А
ґ
main/dense_4/kernel/Adam
VariableV2*
dtype0*
_output_shapes
:	А*
shared_name *&
_class
loc:@main/dense_4/kernel*
	container *
shape:	А
к
main/dense_4/kernel/Adam/AssignAssignmain/dense_4/kernel/Adam*main/dense_4/kernel/Adam/Initializer/zeros*
use_locking(*
T0*&
_class
loc:@main/dense_4/kernel*
validate_shape(*
_output_shapes
:	А
Х
main/dense_4/kernel/Adam/readIdentitymain/dense_4/kernel/Adam*
_output_shapes
:	А*
T0*&
_class
loc:@main/dense_4/kernel
Ђ
,main/dense_4/kernel/Adam_1/Initializer/zerosConst*&
_class
loc:@main/dense_4/kernel*
valueB	А*    *
dtype0*
_output_shapes
:	А
Є
main/dense_4/kernel/Adam_1
VariableV2*
shape:	А*
dtype0*
_output_shapes
:	А*
shared_name *&
_class
loc:@main/dense_4/kernel*
	container 
р
!main/dense_4/kernel/Adam_1/AssignAssignmain/dense_4/kernel/Adam_1,main/dense_4/kernel/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:	А*
use_locking(*
T0*&
_class
loc:@main/dense_4/kernel
Щ
main/dense_4/kernel/Adam_1/readIdentitymain/dense_4/kernel/Adam_1*
_output_shapes
:	А*
T0*&
_class
loc:@main/dense_4/kernel
Ы
(main/dense_4/bias/Adam/Initializer/zerosConst*$
_class
loc:@main/dense_4/bias*
valueB*    *
dtype0*
_output_shapes
:
®
main/dense_4/bias/Adam
VariableV2*$
_class
loc:@main/dense_4/bias*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
Ё
main/dense_4/bias/Adam/AssignAssignmain/dense_4/bias/Adam(main/dense_4/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*$
_class
loc:@main/dense_4/bias
К
main/dense_4/bias/Adam/readIdentitymain/dense_4/bias/Adam*
T0*$
_class
loc:@main/dense_4/bias*
_output_shapes
:
Э
*main/dense_4/bias/Adam_1/Initializer/zerosConst*$
_class
loc:@main/dense_4/bias*
valueB*    *
dtype0*
_output_shapes
:
™
main/dense_4/bias/Adam_1
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *$
_class
loc:@main/dense_4/bias*
	container 
г
main/dense_4/bias/Adam_1/AssignAssignmain/dense_4/bias/Adam_1*main/dense_4/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@main/dense_4/bias*
validate_shape(*
_output_shapes
:
О
main/dense_4/bias/Adam_1/readIdentitymain/dense_4/bias/Adam_1*
T0*$
_class
loc:@main/dense_4/bias*
_output_shapes
:
W
Adam/learning_rateConst*
valueB
 *Ј—8*
dtype0*
_output_shapes
: 
O

Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
O

Adam/beta2Const*
valueB
 *wЊ?*
dtype0*
_output_shapes
: 
Q
Adam/epsilonConst*
valueB
 *wћ+2*
dtype0*
_output_shapes
: 
Л
'Adam/update_main/dense/kernel/ApplyAdam	ApplyAdammain/dense/kernelmain/dense/kernel/Adammain/dense/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon;gradients/main/dense/MatMul_grad/tuple/control_dependency_1*
T0*$
_class
loc:@main/dense/kernel*
use_nesterov( *
_output_shapes
:	А*
use_locking( 
ю
%Adam/update_main/dense/bias/ApplyAdam	ApplyAdammain/dense/biasmain/dense/bias/Adammain/dense/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon<gradients/main/dense/BiasAdd_grad/tuple/control_dependency_1*
T0*"
_class
loc:@main/dense/bias*
use_nesterov( *
_output_shapes	
:А*
use_locking( 
Ч
)Adam/update_main/dense_1/kernel/ApplyAdam	ApplyAdammain/dense_1/kernelmain/dense_1/kernel/Adammain/dense_1/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon=gradients/main/dense_1/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*&
_class
loc:@main/dense_1/kernel*
use_nesterov( *
_output_shapes
:	@А
К
'Adam/update_main/dense_1/bias/ApplyAdam	ApplyAdammain/dense_1/biasmain/dense_1/bias/Adammain/dense_1/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon>gradients/main/dense_1/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*$
_class
loc:@main/dense_1/bias*
use_nesterov( *
_output_shapes	
:А
Ш
)Adam/update_main/dense_2/kernel/ApplyAdam	ApplyAdammain/dense_2/kernelmain/dense_2/kernel/Adammain/dense_2/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon=gradients/main/dense_2/MatMul_grad/tuple/control_dependency_1*
T0*&
_class
loc:@main/dense_2/kernel*
use_nesterov( * 
_output_shapes
:
АА*
use_locking( 
К
'Adam/update_main/dense_2/bias/ApplyAdam	ApplyAdammain/dense_2/biasmain/dense_2/bias/Adammain/dense_2/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon>gradients/main/dense_2/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*$
_class
loc:@main/dense_2/bias*
use_nesterov( *
_output_shapes	
:А
Ш
)Adam/update_main/dense_3/kernel/ApplyAdam	ApplyAdammain/dense_3/kernelmain/dense_3/kernel/Adammain/dense_3/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon=gradients/main/dense_3/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*&
_class
loc:@main/dense_3/kernel*
use_nesterov( * 
_output_shapes
:
АА
К
'Adam/update_main/dense_3/bias/ApplyAdam	ApplyAdammain/dense_3/biasmain/dense_3/bias/Adammain/dense_3/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon>gradients/main/dense_3/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes	
:А*
use_locking( *
T0*$
_class
loc:@main/dense_3/bias
Ч
)Adam/update_main/dense_4/kernel/ApplyAdam	ApplyAdammain/dense_4/kernelmain/dense_4/kernel/Adammain/dense_4/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon=gradients/main/dense_4/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:	А*
use_locking( *
T0*&
_class
loc:@main/dense_4/kernel
Й
'Adam/update_main/dense_4/bias/ApplyAdam	ApplyAdammain/dense_4/biasmain/dense_4/bias/Adammain/dense_4/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon>gradients/main/dense_4/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*$
_class
loc:@main/dense_4/bias*
use_nesterov( *
_output_shapes
:
Ь
Adam/mulMulbeta1_power/read
Adam/beta1&^Adam/update_main/dense/bias/ApplyAdam(^Adam/update_main/dense/kernel/ApplyAdam(^Adam/update_main/dense_1/bias/ApplyAdam*^Adam/update_main/dense_1/kernel/ApplyAdam(^Adam/update_main/dense_2/bias/ApplyAdam*^Adam/update_main/dense_2/kernel/ApplyAdam(^Adam/update_main/dense_3/bias/ApplyAdam*^Adam/update_main/dense_3/kernel/ApplyAdam(^Adam/update_main/dense_4/bias/ApplyAdam*^Adam/update_main/dense_4/kernel/ApplyAdam*
T0*"
_class
loc:@main/dense/bias*
_output_shapes
: 
Ъ
Adam/AssignAssignbeta1_powerAdam/mul*
validate_shape(*
_output_shapes
: *
use_locking( *
T0*"
_class
loc:@main/dense/bias
Ю

Adam/mul_1Mulbeta2_power/read
Adam/beta2&^Adam/update_main/dense/bias/ApplyAdam(^Adam/update_main/dense/kernel/ApplyAdam(^Adam/update_main/dense_1/bias/ApplyAdam*^Adam/update_main/dense_1/kernel/ApplyAdam(^Adam/update_main/dense_2/bias/ApplyAdam*^Adam/update_main/dense_2/kernel/ApplyAdam(^Adam/update_main/dense_3/bias/ApplyAdam*^Adam/update_main/dense_3/kernel/ApplyAdam(^Adam/update_main/dense_4/bias/ApplyAdam*^Adam/update_main/dense_4/kernel/ApplyAdam*
_output_shapes
: *
T0*"
_class
loc:@main/dense/bias
Ю
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
validate_shape(*
_output_shapes
: *
use_locking( *
T0*"
_class
loc:@main/dense/bias
‘
AdamNoOp^Adam/Assign^Adam/Assign_1&^Adam/update_main/dense/bias/ApplyAdam(^Adam/update_main/dense/kernel/ApplyAdam(^Adam/update_main/dense_1/bias/ApplyAdam*^Adam/update_main/dense_1/kernel/ApplyAdam(^Adam/update_main/dense_2/bias/ApplyAdam*^Adam/update_main/dense_2/kernel/ApplyAdam(^Adam/update_main/dense_3/bias/ApplyAdam*^Adam/update_main/dense_3/kernel/ApplyAdam(^Adam/update_main/dense_4/bias/ApplyAdam*^Adam/update_main/dense_4/kernel/ApplyAdam
Є
AssignAssigntarget/dense/kernelmain/dense/kernel/read*
use_locking(*
T0*&
_class
loc:@target/dense/kernel*
validate_shape(*
_output_shapes
:	А
∞
Assign_1Assigntarget/dense/biasmain/dense/bias/read*
use_locking(*
T0*$
_class
loc:@target/dense/bias*
validate_shape(*
_output_shapes	
:А
ј
Assign_2Assigntarget/dense_1/kernelmain/dense_1/kernel/read*
use_locking(*
T0*(
_class
loc:@target/dense_1/kernel*
validate_shape(*
_output_shapes
:	@А
ґ
Assign_3Assigntarget/dense_1/biasmain/dense_1/bias/read*
use_locking(*
T0*&
_class
loc:@target/dense_1/bias*
validate_shape(*
_output_shapes	
:А
Ѕ
Assign_4Assigntarget/dense_2/kernelmain/dense_2/kernel/read*
use_locking(*
T0*(
_class
loc:@target/dense_2/kernel*
validate_shape(* 
_output_shapes
:
АА
ґ
Assign_5Assigntarget/dense_2/biasmain/dense_2/bias/read*
T0*&
_class
loc:@target/dense_2/bias*
validate_shape(*
_output_shapes	
:А*
use_locking(
Ѕ
Assign_6Assigntarget/dense_3/kernelmain/dense_3/kernel/read*
T0*(
_class
loc:@target/dense_3/kernel*
validate_shape(* 
_output_shapes
:
АА*
use_locking(
ґ
Assign_7Assigntarget/dense_3/biasmain/dense_3/bias/read*
use_locking(*
T0*&
_class
loc:@target/dense_3/bias*
validate_shape(*
_output_shapes	
:А
ј
Assign_8Assigntarget/dense_4/kernelmain/dense_4/kernel/read*
use_locking(*
T0*(
_class
loc:@target/dense_4/kernel*
validate_shape(*
_output_shapes
:	А
µ
Assign_9Assigntarget/dense_4/biasmain/dense_4/bias/read*
T0*&
_class
loc:@target/dense_4/bias*
validate_shape(*
_output_shapes
:*
use_locking(
Т

initNoOp^beta1_power/Assign^beta2_power/Assign^main/dense/bias/Adam/Assign^main/dense/bias/Adam_1/Assign^main/dense/bias/Assign^main/dense/kernel/Adam/Assign ^main/dense/kernel/Adam_1/Assign^main/dense/kernel/Assign^main/dense_1/bias/Adam/Assign ^main/dense_1/bias/Adam_1/Assign^main/dense_1/bias/Assign ^main/dense_1/kernel/Adam/Assign"^main/dense_1/kernel/Adam_1/Assign^main/dense_1/kernel/Assign^main/dense_2/bias/Adam/Assign ^main/dense_2/bias/Adam_1/Assign^main/dense_2/bias/Assign ^main/dense_2/kernel/Adam/Assign"^main/dense_2/kernel/Adam_1/Assign^main/dense_2/kernel/Assign^main/dense_3/bias/Adam/Assign ^main/dense_3/bias/Adam_1/Assign^main/dense_3/bias/Assign ^main/dense_3/kernel/Adam/Assign"^main/dense_3/kernel/Adam_1/Assign^main/dense_3/kernel/Assign^main/dense_4/bias/Adam/Assign ^main/dense_4/bias/Adam_1/Assign^main/dense_4/bias/Assign ^main/dense_4/kernel/Adam/Assign"^main/dense_4/kernel/Adam_1/Assign^main/dense_4/kernel/Assign^target/dense/bias/Assign^target/dense/kernel/Assign^target/dense_1/bias/Assign^target/dense_1/kernel/Assign^target/dense_2/bias/Assign^target/dense_2/kernel/Assign^target/dense_3/bias/Assign^target/dense_3/kernel/Assign^target/dense_4/bias/Assign^target/dense_4/kernel/Assign
R
Placeholder_4Placeholder*
dtype0*
_output_shapes
:*
shape:
R
reward/tagsConst*
dtype0*
_output_shapes
: *
valueB Breward
T
rewardScalarSummaryreward/tagsPlaceholder_4*
T0*
_output_shapes
: 
K
Merge/MergeSummaryMergeSummaryreward*
N*
_output_shapes
: ""
train_op

Adam"ї+
	variables≠+™+
{
main/dense/kernel:0main/dense/kernel/Assignmain/dense/kernel/read:02.main/dense/kernel/Initializer/random_uniform:08
j
main/dense/bias:0main/dense/bias/Assignmain/dense/bias/read:02#main/dense/bias/Initializer/zeros:08
Г
main/dense_1/kernel:0main/dense_1/kernel/Assignmain/dense_1/kernel/read:020main/dense_1/kernel/Initializer/random_uniform:08
r
main/dense_1/bias:0main/dense_1/bias/Assignmain/dense_1/bias/read:02%main/dense_1/bias/Initializer/zeros:08
Г
main/dense_2/kernel:0main/dense_2/kernel/Assignmain/dense_2/kernel/read:020main/dense_2/kernel/Initializer/random_uniform:08
r
main/dense_2/bias:0main/dense_2/bias/Assignmain/dense_2/bias/read:02%main/dense_2/bias/Initializer/zeros:08
Г
main/dense_3/kernel:0main/dense_3/kernel/Assignmain/dense_3/kernel/read:020main/dense_3/kernel/Initializer/random_uniform:08
r
main/dense_3/bias:0main/dense_3/bias/Assignmain/dense_3/bias/read:02%main/dense_3/bias/Initializer/zeros:08
Г
main/dense_4/kernel:0main/dense_4/kernel/Assignmain/dense_4/kernel/read:020main/dense_4/kernel/Initializer/random_uniform:08
r
main/dense_4/bias:0main/dense_4/bias/Assignmain/dense_4/bias/read:02%main/dense_4/bias/Initializer/zeros:08
Г
target/dense/kernel:0target/dense/kernel/Assigntarget/dense/kernel/read:020target/dense/kernel/Initializer/random_uniform:08
r
target/dense/bias:0target/dense/bias/Assigntarget/dense/bias/read:02%target/dense/bias/Initializer/zeros:08
Л
target/dense_1/kernel:0target/dense_1/kernel/Assigntarget/dense_1/kernel/read:022target/dense_1/kernel/Initializer/random_uniform:08
z
target/dense_1/bias:0target/dense_1/bias/Assigntarget/dense_1/bias/read:02'target/dense_1/bias/Initializer/zeros:08
Л
target/dense_2/kernel:0target/dense_2/kernel/Assigntarget/dense_2/kernel/read:022target/dense_2/kernel/Initializer/random_uniform:08
z
target/dense_2/bias:0target/dense_2/bias/Assigntarget/dense_2/bias/read:02'target/dense_2/bias/Initializer/zeros:08
Л
target/dense_3/kernel:0target/dense_3/kernel/Assigntarget/dense_3/kernel/read:022target/dense_3/kernel/Initializer/random_uniform:08
z
target/dense_3/bias:0target/dense_3/bias/Assigntarget/dense_3/bias/read:02'target/dense_3/bias/Initializer/zeros:08
Л
target/dense_4/kernel:0target/dense_4/kernel/Assigntarget/dense_4/kernel/read:022target/dense_4/kernel/Initializer/random_uniform:08
z
target/dense_4/bias:0target/dense_4/bias/Assigntarget/dense_4/bias/read:02'target/dense_4/bias/Initializer/zeros:08
T
beta1_power:0beta1_power/Assignbeta1_power/read:02beta1_power/initial_value:0
T
beta2_power:0beta2_power/Assignbeta2_power/read:02beta2_power/initial_value:0
Д
main/dense/kernel/Adam:0main/dense/kernel/Adam/Assignmain/dense/kernel/Adam/read:02*main/dense/kernel/Adam/Initializer/zeros:0
М
main/dense/kernel/Adam_1:0main/dense/kernel/Adam_1/Assignmain/dense/kernel/Adam_1/read:02,main/dense/kernel/Adam_1/Initializer/zeros:0
|
main/dense/bias/Adam:0main/dense/bias/Adam/Assignmain/dense/bias/Adam/read:02(main/dense/bias/Adam/Initializer/zeros:0
Д
main/dense/bias/Adam_1:0main/dense/bias/Adam_1/Assignmain/dense/bias/Adam_1/read:02*main/dense/bias/Adam_1/Initializer/zeros:0
М
main/dense_1/kernel/Adam:0main/dense_1/kernel/Adam/Assignmain/dense_1/kernel/Adam/read:02,main/dense_1/kernel/Adam/Initializer/zeros:0
Ф
main/dense_1/kernel/Adam_1:0!main/dense_1/kernel/Adam_1/Assign!main/dense_1/kernel/Adam_1/read:02.main/dense_1/kernel/Adam_1/Initializer/zeros:0
Д
main/dense_1/bias/Adam:0main/dense_1/bias/Adam/Assignmain/dense_1/bias/Adam/read:02*main/dense_1/bias/Adam/Initializer/zeros:0
М
main/dense_1/bias/Adam_1:0main/dense_1/bias/Adam_1/Assignmain/dense_1/bias/Adam_1/read:02,main/dense_1/bias/Adam_1/Initializer/zeros:0
М
main/dense_2/kernel/Adam:0main/dense_2/kernel/Adam/Assignmain/dense_2/kernel/Adam/read:02,main/dense_2/kernel/Adam/Initializer/zeros:0
Ф
main/dense_2/kernel/Adam_1:0!main/dense_2/kernel/Adam_1/Assign!main/dense_2/kernel/Adam_1/read:02.main/dense_2/kernel/Adam_1/Initializer/zeros:0
Д
main/dense_2/bias/Adam:0main/den
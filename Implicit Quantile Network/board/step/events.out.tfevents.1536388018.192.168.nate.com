       ЃK"	  ькфжAbrain.Event:22еЯ^Єр     ф}фц	ЋьькфжA"С
n
PlaceholderPlaceholder*
dtype0*'
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
p
Placeholder_1Placeholder*'
_output_shapes
:џџџџџџџџџ *
shape:џџџџџџџџџ *
dtype0
p
Placeholder_2Placeholder*'
_output_shapes
:џџџџџџџџџ *
shape:џџџџџџџџџ *
dtype0
p
Placeholder_3Placeholder*
shape:џџџџџџџџџ*
dtype0*'
_output_shapes
:џџџџџџџџџ
d
main/Tile/multiplesConst*
valueB"       *
dtype0*
_output_shapes
:
x
	main/TileTilePlaceholdermain/Tile/multiples*

Tmultiples0*
T0*(
_output_shapes
:џџџџџџџџџ
c
main/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"џџџџ   
v
main/ReshapeReshape	main/Tilemain/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Љ
2main/dense/kernel/Initializer/random_uniform/shapeConst*$
_class
loc:@main/dense/kernel*
valueB"      *
dtype0*
_output_shapes
:

0main/dense/kernel/Initializer/random_uniform/minConst*$
_class
loc:@main/dense/kernel*
valueB
 *JQZО*
dtype0*
_output_shapes
: 

0main/dense/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *$
_class
loc:@main/dense/kernel*
valueB
 *JQZ>
ѕ
:main/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform2main/dense/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	*

seed *
T0*$
_class
loc:@main/dense/kernel*
seed2 
т
0main/dense/kernel/Initializer/random_uniform/subSub0main/dense/kernel/Initializer/random_uniform/max0main/dense/kernel/Initializer/random_uniform/min*
T0*$
_class
loc:@main/dense/kernel*
_output_shapes
: 
ѕ
0main/dense/kernel/Initializer/random_uniform/mulMul:main/dense/kernel/Initializer/random_uniform/RandomUniform0main/dense/kernel/Initializer/random_uniform/sub*$
_class
loc:@main/dense/kernel*
_output_shapes
:	*
T0
ч
,main/dense/kernel/Initializer/random_uniformAdd0main/dense/kernel/Initializer/random_uniform/mul0main/dense/kernel/Initializer/random_uniform/min*
_output_shapes
:	*
T0*$
_class
loc:@main/dense/kernel
­
main/dense/kernel
VariableV2*
shape:	*
dtype0*
_output_shapes
:	*
shared_name *$
_class
loc:@main/dense/kernel*
	container 
м
main/dense/kernel/AssignAssignmain/dense/kernel,main/dense/kernel/Initializer/random_uniform*
use_locking(*
T0*$
_class
loc:@main/dense/kernel*
validate_shape(*
_output_shapes
:	

main/dense/kernel/readIdentitymain/dense/kernel*
_output_shapes
:	*
T0*$
_class
loc:@main/dense/kernel

!main/dense/bias/Initializer/zerosConst*"
_class
loc:@main/dense/bias*
valueB*    *
dtype0*
_output_shapes	
:
Ё
main/dense/bias
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *"
_class
loc:@main/dense/bias*
	container 
Ч
main/dense/bias/AssignAssignmain/dense/bias!main/dense/bias/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@main/dense/bias*
validate_shape(*
_output_shapes	
:
{
main/dense/bias/readIdentitymain/dense/bias*
_output_shapes	
:*
T0*"
_class
loc:@main/dense/bias

main/dense/MatMulMatMulmain/Reshapemain/dense/kernel/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 

main/dense/BiasAddBiasAddmain/dense/MatMulmain/dense/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:џџџџџџџџџ
^
main/dense/SeluSelumain/dense/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
e
main/Reshape_1/shapeConst*
_output_shapes
:*
valueB"џџџџ   *
dtype0
~
main/Reshape_1ReshapePlaceholder_1main/Reshape_1/shape*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
п

main/ConstConst*
valueB@"    лI@лЩ@фЫAлIAбS{AфЫAпэЏAлЩAж1тAбSћAц:
BфЫBт\#Bпэ/Bн~<BлIBи UBж1bBдТnBбS{BgђBц:BeBфЫBcBт\ЃB`ЅЉBпэЏB^6ЖBн~МB\ЧТBлЩBYXЯBи еBWщлBж1тBUzшBдТюBRѕBбSћB(Ю CgђCЇCц:
C&_CeCЅЇCфЫC#№CcCЂ8 Cт\#C!&C`Ѕ)C Щ,Cпэ/C3C^66CZ9Cн~<CЃ?C\ЧBCыEC*
dtype0*
_output_shapes

:@

main/MatMulMatMulmain/Reshape_1
main/Const*
T0*'
_output_shapes
:џџџџџџџџџ@*
transpose_a( *
transpose_b( 
N
main/CosCosmain/MatMul*
T0*'
_output_shapes
:џџџџџџџџџ@
­
4main/dense_1/kernel/Initializer/random_uniform/shapeConst*&
_class
loc:@main/dense_1/kernel*
valueB"@      *
dtype0*
_output_shapes
:

2main/dense_1/kernel/Initializer/random_uniform/minConst*&
_class
loc:@main/dense_1/kernel*
valueB
 *ѓ5О*
dtype0*
_output_shapes
: 

2main/dense_1/kernel/Initializer/random_uniform/maxConst*&
_class
loc:@main/dense_1/kernel*
valueB
 *ѓ5>*
dtype0*
_output_shapes
: 
ћ
<main/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform4main/dense_1/kernel/Initializer/random_uniform/shape*&
_class
loc:@main/dense_1/kernel*
seed2 *
dtype0*
_output_shapes
:	@*

seed *
T0
ъ
2main/dense_1/kernel/Initializer/random_uniform/subSub2main/dense_1/kernel/Initializer/random_uniform/max2main/dense_1/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@main/dense_1/kernel*
_output_shapes
: 
§
2main/dense_1/kernel/Initializer/random_uniform/mulMul<main/dense_1/kernel/Initializer/random_uniform/RandomUniform2main/dense_1/kernel/Initializer/random_uniform/sub*
T0*&
_class
loc:@main/dense_1/kernel*
_output_shapes
:	@
я
.main/dense_1/kernel/Initializer/random_uniformAdd2main/dense_1/kernel/Initializer/random_uniform/mul2main/dense_1/kernel/Initializer/random_uniform/min*&
_class
loc:@main/dense_1/kernel*
_output_shapes
:	@*
T0
Б
main/dense_1/kernel
VariableV2*
shape:	@*
dtype0*
_output_shapes
:	@*
shared_name *&
_class
loc:@main/dense_1/kernel*
	container 
ф
main/dense_1/kernel/AssignAssignmain/dense_1/kernel.main/dense_1/kernel/Initializer/random_uniform*
use_locking(*
T0*&
_class
loc:@main/dense_1/kernel*
validate_shape(*
_output_shapes
:	@

main/dense_1/kernel/readIdentitymain/dense_1/kernel*
_output_shapes
:	@*
T0*&
_class
loc:@main/dense_1/kernel

#main/dense_1/bias/Initializer/zerosConst*$
_class
loc:@main/dense_1/bias*
valueB*    *
dtype0*
_output_shapes	
:
Ѕ
main/dense_1/bias
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *$
_class
loc:@main/dense_1/bias*
	container 
Я
main/dense_1/bias/AssignAssignmain/dense_1/bias#main/dense_1/bias/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*$
_class
loc:@main/dense_1/bias*
validate_shape(

main/dense_1/bias/readIdentitymain/dense_1/bias*
T0*$
_class
loc:@main/dense_1/bias*
_output_shapes	
:

main/dense_1/MatMulMatMulmain/Cosmain/dense_1/kernel/read*
transpose_b( *
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( 

main/dense_1/BiasAddBiasAddmain/dense_1/MatMulmain/dense_1/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:џџџџџџџџџ
b
main/dense_1/ReluRelumain/dense_1/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
f
main/MulMulmain/dense/Selumain/dense_1/Relu*(
_output_shapes
:џџџџџџџџџ*
T0
­
4main/dense_2/kernel/Initializer/random_uniform/shapeConst*&
_class
loc:@main/dense_2/kernel*
valueB"      *
dtype0*
_output_shapes
:

2main/dense_2/kernel/Initializer/random_uniform/minConst*&
_class
loc:@main/dense_2/kernel*
valueB
 *јKЦН*
dtype0*
_output_shapes
: 

2main/dense_2/kernel/Initializer/random_uniform/maxConst*&
_class
loc:@main/dense_2/kernel*
valueB
 *јKЦ=*
dtype0*
_output_shapes
: 
ќ
<main/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform4main/dense_2/kernel/Initializer/random_uniform/shape*
T0*&
_class
loc:@main/dense_2/kernel*
seed2 *
dtype0* 
_output_shapes
:
*

seed 
ъ
2main/dense_2/kernel/Initializer/random_uniform/subSub2main/dense_2/kernel/Initializer/random_uniform/max2main/dense_2/kernel/Initializer/random_uniform/min*&
_class
loc:@main/dense_2/kernel*
_output_shapes
: *
T0
ў
2main/dense_2/kernel/Initializer/random_uniform/mulMul<main/dense_2/kernel/Initializer/random_uniform/RandomUniform2main/dense_2/kernel/Initializer/random_uniform/sub*
T0*&
_class
loc:@main/dense_2/kernel* 
_output_shapes
:

№
.main/dense_2/kernel/Initializer/random_uniformAdd2main/dense_2/kernel/Initializer/random_uniform/mul2main/dense_2/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@main/dense_2/kernel* 
_output_shapes
:

Г
main/dense_2/kernel
VariableV2*
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *&
_class
loc:@main/dense_2/kernel*
	container 
х
main/dense_2/kernel/AssignAssignmain/dense_2/kernel.main/dense_2/kernel/Initializer/random_uniform*
use_locking(*
T0*&
_class
loc:@main/dense_2/kernel*
validate_shape(* 
_output_shapes
:


main/dense_2/kernel/readIdentitymain/dense_2/kernel*
T0*&
_class
loc:@main/dense_2/kernel* 
_output_shapes
:


#main/dense_2/bias/Initializer/zerosConst*$
_class
loc:@main/dense_2/bias*
valueB*    *
dtype0*
_output_shapes	
:
Ѕ
main/dense_2/bias
VariableV2*
_output_shapes	
:*
shared_name *$
_class
loc:@main/dense_2/bias*
	container *
shape:*
dtype0
Я
main/dense_2/bias/AssignAssignmain/dense_2/bias#main/dense_2/bias/Initializer/zeros*$
_class
loc:@main/dense_2/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0

main/dense_2/bias/readIdentitymain/dense_2/bias*
_output_shapes	
:*
T0*$
_class
loc:@main/dense_2/bias

main/dense_2/MatMulMatMulmain/Mulmain/dense_2/kernel/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 

main/dense_2/BiasAddBiasAddmain/dense_2/MatMulmain/dense_2/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:џџџџџџџџџ
b
main/dense_2/ReluRelumain/dense_2/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
­
4main/dense_3/kernel/Initializer/random_uniform/shapeConst*&
_class
loc:@main/dense_3/kernel*
valueB"      *
dtype0*
_output_shapes
:

2main/dense_3/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *&
_class
loc:@main/dense_3/kernel*
valueB
 *јKЦН

2main/dense_3/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *&
_class
loc:@main/dense_3/kernel*
valueB
 *јKЦ=
ќ
<main/dense_3/kernel/Initializer/random_uniform/RandomUniformRandomUniform4main/dense_3/kernel/Initializer/random_uniform/shape*
T0*&
_class
loc:@main/dense_3/kernel*
seed2 *
dtype0* 
_output_shapes
:
*

seed 
ъ
2main/dense_3/kernel/Initializer/random_uniform/subSub2main/dense_3/kernel/Initializer/random_uniform/max2main/dense_3/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@main/dense_3/kernel*
_output_shapes
: 
ў
2main/dense_3/kernel/Initializer/random_uniform/mulMul<main/dense_3/kernel/Initializer/random_uniform/RandomUniform2main/dense_3/kernel/Initializer/random_uniform/sub*
T0*&
_class
loc:@main/dense_3/kernel* 
_output_shapes
:

№
.main/dense_3/kernel/Initializer/random_uniformAdd2main/dense_3/kernel/Initializer/random_uniform/mul2main/dense_3/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@main/dense_3/kernel* 
_output_shapes
:

Г
main/dense_3/kernel
VariableV2*&
_class
loc:@main/dense_3/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name 
х
main/dense_3/kernel/AssignAssignmain/dense_3/kernel.main/dense_3/kernel/Initializer/random_uniform*
T0*&
_class
loc:@main/dense_3/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(

main/dense_3/kernel/readIdentitymain/dense_3/kernel*
T0*&
_class
loc:@main/dense_3/kernel* 
_output_shapes
:


#main/dense_3/bias/Initializer/zerosConst*$
_class
loc:@main/dense_3/bias*
valueB*    *
dtype0*
_output_shapes	
:
Ѕ
main/dense_3/bias
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *$
_class
loc:@main/dense_3/bias*
	container *
shape:
Я
main/dense_3/bias/AssignAssignmain/dense_3/bias#main/dense_3/bias/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@main/dense_3/bias*
validate_shape(*
_output_shapes	
:

main/dense_3/bias/readIdentitymain/dense_3/bias*
_output_shapes	
:*
T0*$
_class
loc:@main/dense_3/bias
Ѓ
main/dense_3/MatMulMatMulmain/dense_2/Relumain/dense_3/kernel/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 

main/dense_3/BiasAddBiasAddmain/dense_3/MatMulmain/dense_3/bias/read*
data_formatNHWC*(
_output_shapes
:џџџџџџџџџ*
T0
b
main/dense_3/ReluRelumain/dense_3/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
­
4main/dense_4/kernel/Initializer/random_uniform/shapeConst*&
_class
loc:@main/dense_4/kernel*
valueB"      *
dtype0*
_output_shapes
:

2main/dense_4/kernel/Initializer/random_uniform/minConst*&
_class
loc:@main/dense_4/kernel*
valueB
 *§[О*
dtype0*
_output_shapes
: 

2main/dense_4/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *&
_class
loc:@main/dense_4/kernel*
valueB
 *§[>
ћ
<main/dense_4/kernel/Initializer/random_uniform/RandomUniformRandomUniform4main/dense_4/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	*

seed *
T0*&
_class
loc:@main/dense_4/kernel*
seed2 
ъ
2main/dense_4/kernel/Initializer/random_uniform/subSub2main/dense_4/kernel/Initializer/random_uniform/max2main/dense_4/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@main/dense_4/kernel*
_output_shapes
: 
§
2main/dense_4/kernel/Initializer/random_uniform/mulMul<main/dense_4/kernel/Initializer/random_uniform/RandomUniform2main/dense_4/kernel/Initializer/random_uniform/sub*
T0*&
_class
loc:@main/dense_4/kernel*
_output_shapes
:	
я
.main/dense_4/kernel/Initializer/random_uniformAdd2main/dense_4/kernel/Initializer/random_uniform/mul2main/dense_4/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@main/dense_4/kernel*
_output_shapes
:	
Б
main/dense_4/kernel
VariableV2*
dtype0*
_output_shapes
:	*
shared_name *&
_class
loc:@main/dense_4/kernel*
	container *
shape:	
ф
main/dense_4/kernel/AssignAssignmain/dense_4/kernel.main/dense_4/kernel/Initializer/random_uniform*&
_class
loc:@main/dense_4/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0

main/dense_4/kernel/readIdentitymain/dense_4/kernel*
T0*&
_class
loc:@main/dense_4/kernel*
_output_shapes
:	

#main/dense_4/bias/Initializer/zerosConst*
_output_shapes
:*$
_class
loc:@main/dense_4/bias*
valueB*    *
dtype0
Ѓ
main/dense_4/bias
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *$
_class
loc:@main/dense_4/bias
Ю
main/dense_4/bias/AssignAssignmain/dense_4/bias#main/dense_4/bias/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*$
_class
loc:@main/dense_4/bias*
validate_shape(

main/dense_4/bias/readIdentitymain/dense_4/bias*
T0*$
_class
loc:@main/dense_4/bias*
_output_shapes
:
Ђ
main/dense_4/MatMulMatMulmain/dense_3/Relumain/dense_4/kernel/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 

main/dense_4/BiasAddBiasAddmain/dense_4/MatMulmain/dense_4/bias/read*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ*
T0
N
main/Const_1Const*
dtype0*
_output_shapes
: *
value	B : 
V
main/split/split_dimConst*
value	B : *
dtype0*
_output_shapes
: 
в

main/splitSplitmain/split/split_dimmain/dense_4/BiasAdd*
T0*і
_output_shapesу
р:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split 
З
main/transpose/xPack
main/splitmain/split:1main/split:2main/split:3main/split:4main/split:5main/split:6main/split:7main/split:8main/split:9main/split:10main/split:11main/split:12main/split:13main/split:14main/split:15main/split:16main/split:17main/split:18main/split:19main/split:20main/split:21main/split:22main/split:23main/split:24main/split:25main/split:26main/split:27main/split:28main/split:29main/split:30main/split:31*
T0*

axis *
N *+
_output_shapes
: џџџџџџџџџ
h
main/transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:

main/transpose	Transposemain/transpose/xmain/transpose/perm*
T0*+
_output_shapes
: џџџџџџџџџ*
Tperm0
f
target/Tile/multiplesConst*
valueB"       *
dtype0*
_output_shapes
:
|
target/TileTilePlaceholdertarget/Tile/multiples*
T0*(
_output_shapes
:џџџџџџџџџ*

Tmultiples0
e
target/Reshape/shapeConst*
valueB"џџџџ   *
dtype0*
_output_shapes
:
|
target/ReshapeReshapetarget/Tiletarget/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
­
4target/dense/kernel/Initializer/random_uniform/shapeConst*&
_class
loc:@target/dense/kernel*
valueB"      *
dtype0*
_output_shapes
:

2target/dense/kernel/Initializer/random_uniform/minConst*&
_class
loc:@target/dense/kernel*
valueB
 *JQZО*
dtype0*
_output_shapes
: 

2target/dense/kernel/Initializer/random_uniform/maxConst*&
_class
loc:@target/dense/kernel*
valueB
 *JQZ>*
dtype0*
_output_shapes
: 
ћ
<target/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform4target/dense/kernel/Initializer/random_uniform/shape*&
_class
loc:@target/dense/kernel*
seed2 *
dtype0*
_output_shapes
:	*

seed *
T0
ъ
2target/dense/kernel/Initializer/random_uniform/subSub2target/dense/kernel/Initializer/random_uniform/max2target/dense/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@target/dense/kernel*
_output_shapes
: 
§
2target/dense/kernel/Initializer/random_uniform/mulMul<target/dense/kernel/Initializer/random_uniform/RandomUniform2target/dense/kernel/Initializer/random_uniform/sub*
_output_shapes
:	*
T0*&
_class
loc:@target/dense/kernel
я
.target/dense/kernel/Initializer/random_uniformAdd2target/dense/kernel/Initializer/random_uniform/mul2target/dense/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@target/dense/kernel*
_output_shapes
:	
Б
target/dense/kernel
VariableV2*
dtype0*
_output_shapes
:	*
shared_name *&
_class
loc:@target/dense/kernel*
	container *
shape:	
ф
target/dense/kernel/AssignAssigntarget/dense/kernel.target/dense/kernel/Initializer/random_uniform*
use_locking(*
T0*&
_class
loc:@target/dense/kernel*
validate_shape(*
_output_shapes
:	

target/dense/kernel/readIdentitytarget/dense/kernel*&
_class
loc:@target/dense/kernel*
_output_shapes
:	*
T0

#target/dense/bias/Initializer/zerosConst*
_output_shapes	
:*$
_class
loc:@target/dense/bias*
valueB*    *
dtype0
Ѕ
target/dense/bias
VariableV2*
_output_shapes	
:*
shared_name *$
_class
loc:@target/dense/bias*
	container *
shape:*
dtype0
Я
target/dense/bias/AssignAssigntarget/dense/bias#target/dense/bias/Initializer/zeros*
T0*$
_class
loc:@target/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(

target/dense/bias/readIdentitytarget/dense/bias*
_output_shapes	
:*
T0*$
_class
loc:@target/dense/bias
 
target/dense/MatMulMatMultarget/Reshapetarget/dense/kernel/read*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0

target/dense/BiasAddBiasAddtarget/dense/MatMultarget/dense/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:џџџџџџџџџ
b
target/dense/SeluSelutarget/dense/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
g
target/Reshape_1/shapeConst*
valueB"џџџџ   *
dtype0*
_output_shapes
:

target/Reshape_1ReshapePlaceholder_1target/Reshape_1/shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
с
target/ConstConst*
valueB@"    лI@лЩ@фЫAлIAбS{AфЫAпэЏAлЩAж1тAбSћAц:
BфЫBт\#Bпэ/Bн~<BлIBи UBж1bBдТnBбS{BgђBц:BeBфЫBcBт\ЃB`ЅЉBпэЏB^6ЖBн~МB\ЧТBлЩBYXЯBи еBWщлBж1тBUzшBдТюBRѕBбSћB(Ю CgђCЇCц:
C&_CeCЅЇCфЫC#№CcCЂ8 Cт\#C!&C`Ѕ)C Щ,Cпэ/C3C^66CZ9Cн~<CЃ?C\ЧBCыEC*
dtype0*
_output_shapes

:@

target/MatMulMatMultarget/Reshape_1target/Const*
transpose_b( *
T0*'
_output_shapes
:џџџџџџџџџ@*
transpose_a( 
R

target/CosCostarget/MatMul*
T0*'
_output_shapes
:џџџџџџџџџ@
Б
6target/dense_1/kernel/Initializer/random_uniform/shapeConst*(
_class
loc:@target/dense_1/kernel*
valueB"@      *
dtype0*
_output_shapes
:
Ѓ
4target/dense_1/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *(
_class
loc:@target/dense_1/kernel*
valueB
 *ѓ5О*
dtype0
Ѓ
4target/dense_1/kernel/Initializer/random_uniform/maxConst*(
_class
loc:@target/dense_1/kernel*
valueB
 *ѓ5>*
dtype0*
_output_shapes
: 

>target/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform6target/dense_1/kernel/Initializer/random_uniform/shape*

seed *
T0*(
_class
loc:@target/dense_1/kernel*
seed2 *
dtype0*
_output_shapes
:	@
ђ
4target/dense_1/kernel/Initializer/random_uniform/subSub4target/dense_1/kernel/Initializer/random_uniform/max4target/dense_1/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@target/dense_1/kernel*
_output_shapes
: 

4target/dense_1/kernel/Initializer/random_uniform/mulMul>target/dense_1/kernel/Initializer/random_uniform/RandomUniform4target/dense_1/kernel/Initializer/random_uniform/sub*
_output_shapes
:	@*
T0*(
_class
loc:@target/dense_1/kernel
ї
0target/dense_1/kernel/Initializer/random_uniformAdd4target/dense_1/kernel/Initializer/random_uniform/mul4target/dense_1/kernel/Initializer/random_uniform/min*(
_class
loc:@target/dense_1/kernel*
_output_shapes
:	@*
T0
Е
target/dense_1/kernel
VariableV2*
shared_name *(
_class
loc:@target/dense_1/kernel*
	container *
shape:	@*
dtype0*
_output_shapes
:	@
ь
target/dense_1/kernel/AssignAssigntarget/dense_1/kernel0target/dense_1/kernel/Initializer/random_uniform*
validate_shape(*
_output_shapes
:	@*
use_locking(*
T0*(
_class
loc:@target/dense_1/kernel

target/dense_1/kernel/readIdentitytarget/dense_1/kernel*
_output_shapes
:	@*
T0*(
_class
loc:@target/dense_1/kernel

%target/dense_1/bias/Initializer/zerosConst*&
_class
loc:@target/dense_1/bias*
valueB*    *
dtype0*
_output_shapes	
:
Љ
target/dense_1/bias
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *&
_class
loc:@target/dense_1/bias*
	container *
shape:
з
target/dense_1/bias/AssignAssigntarget/dense_1/bias%target/dense_1/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*&
_class
loc:@target/dense_1/bias

target/dense_1/bias/readIdentitytarget/dense_1/bias*
T0*&
_class
loc:@target/dense_1/bias*
_output_shapes	
:
 
target/dense_1/MatMulMatMul
target/Costarget/dense_1/kernel/read*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0

target/dense_1/BiasAddBiasAddtarget/dense_1/MatMultarget/dense_1/bias/read*
data_formatNHWC*(
_output_shapes
:џџџџџџџџџ*
T0
f
target/dense_1/ReluRelutarget/dense_1/BiasAdd*(
_output_shapes
:џџџџџџџџџ*
T0
l

target/MulMultarget/dense/Selutarget/dense_1/Relu*(
_output_shapes
:џџџџџџџџџ*
T0
Б
6target/dense_2/kernel/Initializer/random_uniform/shapeConst*(
_class
loc:@target/dense_2/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ѓ
4target/dense_2/kernel/Initializer/random_uniform/minConst*(
_class
loc:@target/dense_2/kernel*
valueB
 *јKЦН*
dtype0*
_output_shapes
: 
Ѓ
4target/dense_2/kernel/Initializer/random_uniform/maxConst*(
_class
loc:@target/dense_2/kernel*
valueB
 *јKЦ=*
dtype0*
_output_shapes
: 

>target/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform6target/dense_2/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
*

seed *
T0*(
_class
loc:@target/dense_2/kernel*
seed2 
ђ
4target/dense_2/kernel/Initializer/random_uniform/subSub4target/dense_2/kernel/Initializer/random_uniform/max4target/dense_2/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@target/dense_2/kernel*
_output_shapes
: 

4target/dense_2/kernel/Initializer/random_uniform/mulMul>target/dense_2/kernel/Initializer/random_uniform/RandomUniform4target/dense_2/kernel/Initializer/random_uniform/sub*(
_class
loc:@target/dense_2/kernel* 
_output_shapes
:
*
T0
ј
0target/dense_2/kernel/Initializer/random_uniformAdd4target/dense_2/kernel/Initializer/random_uniform/mul4target/dense_2/kernel/Initializer/random_uniform/min*(
_class
loc:@target/dense_2/kernel* 
_output_shapes
:
*
T0
З
target/dense_2/kernel
VariableV2*
shared_name *(
_class
loc:@target/dense_2/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:

э
target/dense_2/kernel/AssignAssigntarget/dense_2/kernel0target/dense_2/kernel/Initializer/random_uniform*
T0*(
_class
loc:@target/dense_2/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(

target/dense_2/kernel/readIdentitytarget/dense_2/kernel*(
_class
loc:@target/dense_2/kernel* 
_output_shapes
:
*
T0

%target/dense_2/bias/Initializer/zerosConst*&
_class
loc:@target/dense_2/bias*
valueB*    *
dtype0*
_output_shapes	
:
Љ
target/dense_2/bias
VariableV2*
shared_name *&
_class
loc:@target/dense_2/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
з
target/dense_2/bias/AssignAssigntarget/dense_2/bias%target/dense_2/bias/Initializer/zeros*
use_locking(*
T0*&
_class
loc:@target/dense_2/bias*
validate_shape(*
_output_shapes	
:

target/dense_2/bias/readIdentitytarget/dense_2/bias*
T0*&
_class
loc:@target/dense_2/bias*
_output_shapes	
:
 
target/dense_2/MatMulMatMul
target/Multarget/dense_2/kernel/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 

target/dense_2/BiasAddBiasAddtarget/dense_2/MatMultarget/dense_2/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:џџџџџџџџџ
f
target/dense_2/ReluRelutarget/dense_2/BiasAdd*(
_output_shapes
:џџџџџџџџџ*
T0
Б
6target/dense_3/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*(
_class
loc:@target/dense_3/kernel*
valueB"      *
dtype0
Ѓ
4target/dense_3/kernel/Initializer/random_uniform/minConst*(
_class
loc:@target/dense_3/kernel*
valueB
 *јKЦН*
dtype0*
_output_shapes
: 
Ѓ
4target/dense_3/kernel/Initializer/random_uniform/maxConst*(
_class
loc:@target/dense_3/kernel*
valueB
 *јKЦ=*
dtype0*
_output_shapes
: 

>target/dense_3/kernel/Initializer/random_uniform/RandomUniformRandomUniform6target/dense_3/kernel/Initializer/random_uniform/shape* 
_output_shapes
:
*

seed *
T0*(
_class
loc:@target/dense_3/kernel*
seed2 *
dtype0
ђ
4target/dense_3/kernel/Initializer/random_uniform/subSub4target/dense_3/kernel/Initializer/random_uniform/max4target/dense_3/kernel/Initializer/random_uniform/min*(
_class
loc:@target/dense_3/kernel*
_output_shapes
: *
T0

4target/dense_3/kernel/Initializer/random_uniform/mulMul>target/dense_3/kernel/Initializer/random_uniform/RandomUniform4target/dense_3/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
*
T0*(
_class
loc:@target/dense_3/kernel
ј
0target/dense_3/kernel/Initializer/random_uniformAdd4target/dense_3/kernel/Initializer/random_uniform/mul4target/dense_3/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@target/dense_3/kernel* 
_output_shapes
:

З
target/dense_3/kernel
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *(
_class
loc:@target/dense_3/kernel*
	container *
shape:

э
target/dense_3/kernel/AssignAssigntarget/dense_3/kernel0target/dense_3/kernel/Initializer/random_uniform*
T0*(
_class
loc:@target/dense_3/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(

target/dense_3/kernel/readIdentitytarget/dense_3/kernel* 
_output_shapes
:
*
T0*(
_class
loc:@target/dense_3/kernel

%target/dense_3/bias/Initializer/zerosConst*&
_class
loc:@target/dense_3/bias*
valueB*    *
dtype0*
_output_shapes	
:
Љ
target/dense_3/bias
VariableV2*&
_class
loc:@target/dense_3/bias*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
з
target/dense_3/bias/AssignAssigntarget/dense_3/bias%target/dense_3/bias/Initializer/zeros*
use_locking(*
T0*&
_class
loc:@target/dense_3/bias*
validate_shape(*
_output_shapes	
:

target/dense_3/bias/readIdentitytarget/dense_3/bias*
T0*&
_class
loc:@target/dense_3/bias*
_output_shapes	
:
Љ
target/dense_3/MatMulMatMultarget/dense_2/Relutarget/dense_3/kernel/read*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0

target/dense_3/BiasAddBiasAddtarget/dense_3/MatMultarget/dense_3/bias/read*
data_formatNHWC*(
_output_shapes
:џџџџџџџџџ*
T0
f
target/dense_3/ReluRelutarget/dense_3/BiasAdd*(
_output_shapes
:џџџџџџџџџ*
T0
Б
6target/dense_4/kernel/Initializer/random_uniform/shapeConst*(
_class
loc:@target/dense_4/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ѓ
4target/dense_4/kernel/Initializer/random_uniform/minConst*(
_class
loc:@target/dense_4/kernel*
valueB
 *§[О*
dtype0*
_output_shapes
: 
Ѓ
4target/dense_4/kernel/Initializer/random_uniform/maxConst*(
_class
loc:@target/dense_4/kernel*
valueB
 *§[>*
dtype0*
_output_shapes
: 

>target/dense_4/kernel/Initializer/random_uniform/RandomUniformRandomUniform6target/dense_4/kernel/Initializer/random_uniform/shape*
T0*(
_class
loc:@target/dense_4/kernel*
seed2 *
dtype0*
_output_shapes
:	*

seed 
ђ
4target/dense_4/kernel/Initializer/random_uniform/subSub4target/dense_4/kernel/Initializer/random_uniform/max4target/dense_4/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@target/dense_4/kernel*
_output_shapes
: 

4target/dense_4/kernel/Initializer/random_uniform/mulMul>target/dense_4/kernel/Initializer/random_uniform/RandomUniform4target/dense_4/kernel/Initializer/random_uniform/sub*
T0*(
_class
loc:@target/dense_4/kernel*
_output_shapes
:	
ї
0target/dense_4/kernel/Initializer/random_uniformAdd4target/dense_4/kernel/Initializer/random_uniform/mul4target/dense_4/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@target/dense_4/kernel*
_output_shapes
:	
Е
target/dense_4/kernel
VariableV2*
shared_name *(
_class
loc:@target/dense_4/kernel*
	container *
shape:	*
dtype0*
_output_shapes
:	
ь
target/dense_4/kernel/AssignAssigntarget/dense_4/kernel0target/dense_4/kernel/Initializer/random_uniform*
use_locking(*
T0*(
_class
loc:@target/dense_4/kernel*
validate_shape(*
_output_shapes
:	

target/dense_4/kernel/readIdentitytarget/dense_4/kernel*
T0*(
_class
loc:@target/dense_4/kernel*
_output_shapes
:	

%target/dense_4/bias/Initializer/zerosConst*&
_class
loc:@target/dense_4/bias*
valueB*    *
dtype0*
_output_shapes
:
Ї
target/dense_4/bias
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *&
_class
loc:@target/dense_4/bias*
	container 
ж
target/dense_4/bias/AssignAssigntarget/dense_4/bias%target/dense_4/bias/Initializer/zeros*
use_locking(*
T0*&
_class
loc:@target/dense_4/bias*
validate_shape(*
_output_shapes
:

target/dense_4/bias/readIdentitytarget/dense_4/bias*
T0*&
_class
loc:@target/dense_4/bias*
_output_shapes
:
Ј
target/dense_4/MatMulMatMultarget/dense_3/Relutarget/dense_4/kernel/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 

target/dense_4/BiasAddBiasAddtarget/dense_4/MatMultarget/dense_4/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ
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
и
target/splitSplittarget/split/split_dimtarget/dense_4/BiasAdd*
T0*і
_output_shapesу
р:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split 
љ
target/transpose/xPacktarget/splittarget/split:1target/split:2target/split:3target/split:4target/split:5target/split:6target/split:7target/split:8target/split:9target/split:10target/split:11target/split:12target/split:13target/split:14target/split:15target/split:16target/split:17target/split:18target/split:19target/split:20target/split:21target/split:22target/split:23target/split:24target/split:25target/split:26target/split:27target/split:28target/split:29target/split:30target/split:31*
T0*

axis *
N *+
_output_shapes
: џџџџџџџџџ
j
target/transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:

target/transpose	Transposetarget/transpose/xtarget/transpose/perm*+
_output_shapes
: џџџџџџџџџ*
Tperm0*
T0
Y
ExpandDims/dimConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
y

ExpandDims
ExpandDimsPlaceholder_3ExpandDims/dim*+
_output_shapes
:џџџџџџџџџ*

Tdim0*
T0
\
mulMulmain/transpose
ExpandDims*+
_output_shapes
: џџџџџџџџџ*
T0
W
Sum/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
u
SumSummulSum/reduction_indices*
	keep_dims( *

Tidx0*
T0*'
_output_shapes
: џџџџџџџџџ
R
huber_loss/SubSubSumPlaceholder_2*
_output_shapes

:  *
T0
N
huber_loss/AbsAbshuber_loss/Sub*
_output_shapes

:  *
T0
Y
huber_loss/Minimum/yConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
l
huber_loss/MinimumMinimumhuber_loss/Abshuber_loss/Minimum/y*
T0*
_output_shapes

:  
d
huber_loss/Sub_1Subhuber_loss/Abshuber_loss/Minimum*
T0*
_output_shapes

:  
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

:  *
T0
b
huber_loss/Mul_1Mulhuber_loss/Consthuber_loss/Mul*
T0*
_output_shapes

:  
W
huber_loss/Mul_2/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
huber_loss/Mul_2Mulhuber_loss/Mul_2/xhuber_loss/Sub_1*
T0*
_output_shapes

:  
b
huber_loss/AddAddhuber_loss/Mul_1huber_loss/Mul_2*
T0*
_output_shapes

:  
l
'huber_loss/assert_broadcastable/weightsConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
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
,huber_loss/assert_broadcastable/values/shapeConst*
_output_shapes
:*
valueB"        *
dtype0
m
+huber_loss/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
C
;huber_loss/assert_broadcastable/static_scalar_check_successNoOp

huber_loss/ToFloat_3/xConst<^huber_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *  ?*
dtype0*
_output_shapes
: 
h
huber_loss/Mul_3Mulhuber_loss/Addhuber_loss/ToFloat_3/x*
T0*
_output_shapes

:  
J
sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
R
subSubsub/xPlaceholder_1*'
_output_shapes
:џџџџџџџџџ *
T0
I
sub_1SubPlaceholder_2Sum*
T0*
_output_shapes

:  
K
Less/yConst*
dtype0*
_output_shapes
: *
valueB
 *    
D
LessLesssub_1Less/y*
_output_shapes

:  *
T0
L
mul_1Mulsubhuber_loss/Mul_3*
_output_shapes

:  *
T0
V
mul_2MulPlaceholder_1huber_loss/Mul_3*
T0*
_output_shapes

:  
M
SelectSelectLessmul_1mul_2*
T0*
_output_shapes

:  
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
MeanMeanSum_1Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
R
gradients/ShapeConst*
_output_shapes
: *
valueB *
dtype0
X
gradients/grad_ys_0Const*
valueB
 *  ?*
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

gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
_output_shapes
:*
T0*
Tshape0
c
gradients/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:

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

gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Const_1*
_output_shapes
: *
T0
k
gradients/Sum_1_grad/ShapeConst*
_output_shapes
:*
valueB"        *
dtype0

gradients/Sum_1_grad/SizeConst*
_output_shapes
: *-
_class#
!loc:@gradients/Sum_1_grad/Shape*
value	B :*
dtype0
Ѓ
gradients/Sum_1_grad/addAddSum_1/reduction_indicesgradients/Sum_1_grad/Size*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
_output_shapes
: *
T0
Љ
gradients/Sum_1_grad/modFloorModgradients/Sum_1_grad/addgradients/Sum_1_grad/Size*
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
_output_shapes
: 

gradients/Sum_1_grad/Shape_1Const*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
valueB *
dtype0*
_output_shapes
: 

 gradients/Sum_1_grad/range/startConst*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
value	B : *
dtype0*
_output_shapes
: 

 gradients/Sum_1_grad/range/deltaConst*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
й
gradients/Sum_1_grad/rangeRange gradients/Sum_1_grad/range/startgradients/Sum_1_grad/Size gradients/Sum_1_grad/range/delta*

Tidx0*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
_output_shapes
:

gradients/Sum_1_grad/Fill/valueConst*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
Т
gradients/Sum_1_grad/FillFillgradients/Sum_1_grad/Shape_1gradients/Sum_1_grad/Fill/value*
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape*

index_type0*
_output_shapes
: 

"gradients/Sum_1_grad/DynamicStitchDynamicStitchgradients/Sum_1_grad/rangegradients/Sum_1_grad/modgradients/Sum_1_grad/Shapegradients/Sum_1_grad/Fill*
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
N*#
_output_shapes
:џџџџџџџџџ

gradients/Sum_1_grad/Maximum/yConst*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
Ш
gradients/Sum_1_grad/MaximumMaximum"gradients/Sum_1_grad/DynamicStitchgradients/Sum_1_grad/Maximum/y*
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape*#
_output_shapes
:џџџџџџџџџ
З
gradients/Sum_1_grad/floordivFloorDivgradients/Sum_1_grad/Shapegradients/Sum_1_grad/Maximum*
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
_output_shapes
:

gradients/Sum_1_grad/ReshapeReshapegradients/Mean_grad/truediv"gradients/Sum_1_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:

gradients/Sum_1_grad/TileTilegradients/Sum_1_grad/Reshapegradients/Sum_1_grad/floordiv*

Tmultiples0*
T0*
_output_shapes

:  

0gradients/Select_grad/zeros_like/shape_as_tensorConst*
valueB"        *
dtype0*
_output_shapes
:
k
&gradients/Select_grad/zeros_like/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Н
 gradients/Select_grad/zeros_likeFill0gradients/Select_grad/zeros_like/shape_as_tensor&gradients/Select_grad/zeros_like/Const*
_output_shapes

:  *
T0*

index_type0

gradients/Select_grad/SelectSelectLessgradients/Sum_1_grad/Tile gradients/Select_grad/zeros_like*
T0*
_output_shapes

:  

gradients/Select_grad/Select_1SelectLess gradients/Select_grad/zeros_likegradients/Sum_1_grad/Tile*
T0*
_output_shapes

:  
n
&gradients/Select_grad/tuple/group_depsNoOp^gradients/Select_grad/Select^gradients/Select_grad/Select_1
л
.gradients/Select_grad/tuple/control_dependencyIdentitygradients/Select_grad/Select'^gradients/Select_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/Select_grad/Select*
_output_shapes

:  
с
0gradients/Select_grad/tuple/control_dependency_1Identitygradients/Select_grad/Select_1'^gradients/Select_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/Select_grad/Select_1*
_output_shapes

:  
]
gradients/mul_1_grad/ShapeShapesub*
_output_shapes
:*
T0*
out_type0
m
gradients/mul_1_grad/Shape_1Const*
valueB"        *
dtype0*
_output_shapes
:
К
*gradients/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_1_grad/Shapegradients/mul_1_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0

gradients/mul_1_grad/MulMul.gradients/Select_grad/tuple/control_dependencyhuber_loss/Mul_3*
_output_shapes

:  *
T0
Ѕ
gradients/mul_1_grad/SumSumgradients/mul_1_grad/Mul*gradients/mul_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

gradients/mul_1_grad/ReshapeReshapegradients/mul_1_grad/Sumgradients/mul_1_grad/Shape*
Tshape0*'
_output_shapes
:џџџџџџџџџ *
T0

gradients/mul_1_grad/Mul_1Mulsub.gradients/Select_grad/tuple/control_dependency*
T0*
_output_shapes

:  
Ћ
gradients/mul_1_grad/Sum_1Sumgradients/mul_1_grad/Mul_1,gradients/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

gradients/mul_1_grad/Reshape_1Reshapegradients/mul_1_grad/Sum_1gradients/mul_1_grad/Shape_1*
_output_shapes

:  *
T0*
Tshape0
m
%gradients/mul_1_grad/tuple/group_depsNoOp^gradients/mul_1_grad/Reshape^gradients/mul_1_grad/Reshape_1
т
-gradients/mul_1_grad/tuple/control_dependencyIdentitygradients/mul_1_grad/Reshape&^gradients/mul_1_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/mul_1_grad/Reshape*'
_output_shapes
:џџџџџџџџџ 
п
/gradients/mul_1_grad/tuple/control_dependency_1Identitygradients/mul_1_grad/Reshape_1&^gradients/mul_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/mul_1_grad/Reshape_1*
_output_shapes

:  
g
gradients/mul_2_grad/ShapeShapePlaceholder_1*
_output_shapes
:*
T0*
out_type0
m
gradients/mul_2_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB"        
К
*gradients/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_2_grad/Shapegradients/mul_2_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

gradients/mul_2_grad/MulMul0gradients/Select_grad/tuple/control_dependency_1huber_loss/Mul_3*
T0*
_output_shapes

:  
Ѕ
gradients/mul_2_grad/SumSumgradients/mul_2_grad/Mul*gradients/mul_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

gradients/mul_2_grad/ReshapeReshapegradients/mul_2_grad/Sumgradients/mul_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ 

gradients/mul_2_grad/Mul_1MulPlaceholder_10gradients/Select_grad/tuple/control_dependency_1*
T0*
_output_shapes

:  
Ћ
gradients/mul_2_grad/Sum_1Sumgradients/mul_2_grad/Mul_1,gradients/mul_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

gradients/mul_2_grad/Reshape_1Reshapegradients/mul_2_grad/Sum_1gradients/mul_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:  
m
%gradients/mul_2_grad/tuple/group_depsNoOp^gradients/mul_2_grad/Reshape^gradients/mul_2_grad/Reshape_1
т
-gradients/mul_2_grad/tuple/control_dependencyIdentitygradients/mul_2_grad/Reshape&^gradients/mul_2_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/mul_2_grad/Reshape*'
_output_shapes
:џџџџџџџџџ 
п
/gradients/mul_2_grad/tuple/control_dependency_1Identitygradients/mul_2_grad/Reshape_1&^gradients/mul_2_grad/tuple/group_deps*1
_class'
%#loc:@gradients/mul_2_grad/Reshape_1*
_output_shapes

:  *
T0
н
gradients/AddNAddN/gradients/mul_1_grad/tuple/control_dependency_1/gradients/mul_2_grad/tuple/control_dependency_1*
T0*1
_class'
%#loc:@gradients/mul_1_grad/Reshape_1*
N*
_output_shapes

:  
v
%gradients/huber_loss/Mul_3_grad/ShapeConst*
_output_shapes
:*
valueB"        *
dtype0
j
'gradients/huber_loss/Mul_3_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
л
5gradients/huber_loss/Mul_3_grad/BroadcastGradientArgsBroadcastGradientArgs%gradients/huber_loss/Mul_3_grad/Shape'gradients/huber_loss/Mul_3_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
{
#gradients/huber_loss/Mul_3_grad/MulMulgradients/AddNhuber_loss/ToFloat_3/x*
T0*
_output_shapes

:  
Ц
#gradients/huber_loss/Mul_3_grad/SumSum#gradients/huber_loss/Mul_3_grad/Mul5gradients/huber_loss/Mul_3_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Е
'gradients/huber_loss/Mul_3_grad/ReshapeReshape#gradients/huber_loss/Mul_3_grad/Sum%gradients/huber_loss/Mul_3_grad/Shape*
T0*
Tshape0*
_output_shapes

:  
u
%gradients/huber_loss/Mul_3_grad/Mul_1Mulhuber_loss/Addgradients/AddN*
T0*
_output_shapes

:  
Ь
%gradients/huber_loss/Mul_3_grad/Sum_1Sum%gradients/huber_loss/Mul_3_grad/Mul_17gradients/huber_loss/Mul_3_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Г
)gradients/huber_loss/Mul_3_grad/Reshape_1Reshape%gradients/huber_loss/Mul_3_grad/Sum_1'gradients/huber_loss/Mul_3_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 

0gradients/huber_loss/Mul_3_grad/tuple/group_depsNoOp(^gradients/huber_loss/Mul_3_grad/Reshape*^gradients/huber_loss/Mul_3_grad/Reshape_1

8gradients/huber_loss/Mul_3_grad/tuple/control_dependencyIdentity'gradients/huber_loss/Mul_3_grad/Reshape1^gradients/huber_loss/Mul_3_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/huber_loss/Mul_3_grad/Reshape*
_output_shapes

:  

:gradients/huber_loss/Mul_3_grad/tuple/control_dependency_1Identity)gradients/huber_loss/Mul_3_grad/Reshape_11^gradients/huber_loss/Mul_3_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/huber_loss/Mul_3_grad/Reshape_1*
_output_shapes
: 
q
.gradients/huber_loss/Add_grad/tuple/group_depsNoOp9^gradients/huber_loss/Mul_3_grad/tuple/control_dependency

6gradients/huber_loss/Add_grad/tuple/control_dependencyIdentity8gradients/huber_loss/Mul_3_grad/tuple/control_dependency/^gradients/huber_loss/Add_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/huber_loss/Mul_3_grad/Reshape*
_output_shapes

:  

8gradients/huber_loss/Add_grad/tuple/control_dependency_1Identity8gradients/huber_loss/Mul_3_grad/tuple/control_dependency/^gradients/huber_loss/Add_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/huber_loss/Mul_3_grad/Reshape*
_output_shapes

:  
h
%gradients/huber_loss/Mul_1_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
x
'gradients/huber_loss/Mul_1_grad/Shape_1Const*
valueB"        *
dtype0*
_output_shapes
:
л
5gradients/huber_loss/Mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs%gradients/huber_loss/Mul_1_grad/Shape'gradients/huber_loss/Mul_1_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0

#gradients/huber_loss/Mul_1_grad/MulMul6gradients/huber_loss/Add_grad/tuple/control_dependencyhuber_loss/Mul*
_output_shapes

:  *
T0
Ц
#gradients/huber_loss/Mul_1_grad/SumSum#gradients/huber_loss/Mul_1_grad/Mul5gradients/huber_loss/Mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
­
'gradients/huber_loss/Mul_1_grad/ReshapeReshape#gradients/huber_loss/Mul_1_grad/Sum%gradients/huber_loss/Mul_1_grad/Shape*
_output_shapes
: *
T0*
Tshape0

%gradients/huber_loss/Mul_1_grad/Mul_1Mulhuber_loss/Const6gradients/huber_loss/Add_grad/tuple/control_dependency*
T0*
_output_shapes

:  
Ь
%gradients/huber_loss/Mul_1_grad/Sum_1Sum%gradients/huber_loss/Mul_1_grad/Mul_17gradients/huber_loss/Mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Л
)gradients/huber_loss/Mul_1_grad/Reshape_1Reshape%gradients/huber_loss/Mul_1_grad/Sum_1'gradients/huber_loss/Mul_1_grad/Shape_1*
_output_shapes

:  *
T0*
Tshape0

0gradients/huber_loss/Mul_1_grad/tuple/group_depsNoOp(^gradients/huber_loss/Mul_1_grad/Reshape*^gradients/huber_loss/Mul_1_grad/Reshape_1
§
8gradients/huber_loss/Mul_1_grad/tuple/control_dependencyIdentity'gradients/huber_loss/Mul_1_grad/Reshape1^gradients/huber_loss/Mul_1_grad/tuple/group_deps*
_output_shapes
: *
T0*:
_class0
.,loc:@gradients/huber_loss/Mul_1_grad/Reshape

:gradients/huber_loss/Mul_1_grad/tuple/control_dependency_1Identity)gradients/huber_loss/Mul_1_grad/Reshape_11^gradients/huber_loss/Mul_1_grad/tuple/group_deps*
_output_shapes

:  *
T0*<
_class2
0.loc:@gradients/huber_loss/Mul_1_grad/Reshape_1
h
%gradients/huber_loss/Mul_2_grad/ShapeConst*
_output_shapes
: *
valueB *
dtype0
x
'gradients/huber_loss/Mul_2_grad/Shape_1Const*
valueB"        *
dtype0*
_output_shapes
:
л
5gradients/huber_loss/Mul_2_grad/BroadcastGradientArgsBroadcastGradientArgs%gradients/huber_loss/Mul_2_grad/Shape'gradients/huber_loss/Mul_2_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

#gradients/huber_loss/Mul_2_grad/MulMul8gradients/huber_loss/Add_grad/tuple/control_dependency_1huber_loss/Sub_1*
_output_shapes

:  *
T0
Ц
#gradients/huber_loss/Mul_2_grad/SumSum#gradients/huber_loss/Mul_2_grad/Mul5gradients/huber_loss/Mul_2_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
­
'gradients/huber_loss/Mul_2_grad/ReshapeReshape#gradients/huber_loss/Mul_2_grad/Sum%gradients/huber_loss/Mul_2_grad/Shape*
_output_shapes
: *
T0*
Tshape0
Ѓ
%gradients/huber_loss/Mul_2_grad/Mul_1Mulhuber_loss/Mul_2/x8gradients/huber_loss/Add_grad/tuple/control_dependency_1*
T0*
_output_shapes

:  
Ь
%gradients/huber_loss/Mul_2_grad/Sum_1Sum%gradients/huber_loss/Mul_2_grad/Mul_17gradients/huber_loss/Mul_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Л
)gradients/huber_loss/Mul_2_grad/Reshape_1Reshape%gradients/huber_loss/Mul_2_grad/Sum_1'gradients/huber_loss/Mul_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:  

0gradients/huber_loss/Mul_2_grad/tuple/group_depsNoOp(^gradients/huber_loss/Mul_2_grad/Reshape*^gradients/huber_loss/Mul_2_grad/Reshape_1
§
8gradients/huber_loss/Mul_2_grad/tuple/control_dependencyIdentity'gradients/huber_loss/Mul_2_grad/Reshape1^gradients/huber_loss/Mul_2_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/huber_loss/Mul_2_grad/Reshape*
_output_shapes
: 

:gradients/huber_loss/Mul_2_grad/tuple/control_dependency_1Identity)gradients/huber_loss/Mul_2_grad/Reshape_11^gradients/huber_loss/Mul_2_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/huber_loss/Mul_2_grad/Reshape_1*
_output_shapes

:  
Ё
!gradients/huber_loss/Mul_grad/MulMul:gradients/huber_loss/Mul_1_grad/tuple/control_dependency_1huber_loss/Minimum*
T0*
_output_shapes

:  
Ѓ
#gradients/huber_loss/Mul_grad/Mul_1Mul:gradients/huber_loss/Mul_1_grad/tuple/control_dependency_1huber_loss/Minimum*
T0*
_output_shapes

:  

.gradients/huber_loss/Mul_grad/tuple/group_depsNoOp"^gradients/huber_loss/Mul_grad/Mul$^gradients/huber_loss/Mul_grad/Mul_1
ѕ
6gradients/huber_loss/Mul_grad/tuple/control_dependencyIdentity!gradients/huber_loss/Mul_grad/Mul/^gradients/huber_loss/Mul_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients/huber_loss/Mul_grad/Mul*
_output_shapes

:  
ћ
8gradients/huber_loss/Mul_grad/tuple/control_dependency_1Identity#gradients/huber_loss/Mul_grad/Mul_1/^gradients/huber_loss/Mul_grad/tuple/group_deps*
_output_shapes

:  *
T0*6
_class,
*(loc:@gradients/huber_loss/Mul_grad/Mul_1

#gradients/huber_loss/Sub_1_grad/NegNeg:gradients/huber_loss/Mul_2_grad/tuple/control_dependency_1*
_output_shapes

:  *
T0

0gradients/huber_loss/Sub_1_grad/tuple/group_depsNoOp;^gradients/huber_loss/Mul_2_grad/tuple/control_dependency_1$^gradients/huber_loss/Sub_1_grad/Neg

8gradients/huber_loss/Sub_1_grad/tuple/control_dependencyIdentity:gradients/huber_loss/Mul_2_grad/tuple/control_dependency_11^gradients/huber_loss/Sub_1_grad/tuple/group_deps*
_output_shapes

:  *
T0*<
_class2
0.loc:@gradients/huber_loss/Mul_2_grad/Reshape_1
џ
:gradients/huber_loss/Sub_1_grad/tuple/control_dependency_1Identity#gradients/huber_loss/Sub_1_grad/Neg1^gradients/huber_loss/Sub_1_grad/tuple/group_deps*6
_class,
*(loc:@gradients/huber_loss/Sub_1_grad/Neg*
_output_shapes

:  *
T0
Ў
gradients/AddN_1AddN6gradients/huber_loss/Mul_grad/tuple/control_dependency8gradients/huber_loss/Mul_grad/tuple/control_dependency_1:gradients/huber_loss/Sub_1_grad/tuple/control_dependency_1*
T0*4
_class*
(&loc:@gradients/huber_loss/Mul_grad/Mul*
N*
_output_shapes

:  
x
'gradients/huber_loss/Minimum_grad/ShapeConst*
valueB"        *
dtype0*
_output_shapes
:
l
)gradients/huber_loss/Minimum_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
z
)gradients/huber_loss/Minimum_grad/Shape_2Const*
dtype0*
_output_shapes
:*
valueB"        
r
-gradients/huber_loss/Minimum_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ф
'gradients/huber_loss/Minimum_grad/zerosFill)gradients/huber_loss/Minimum_grad/Shape_2-gradients/huber_loss/Minimum_grad/zeros/Const*
T0*

index_type0*
_output_shapes

:  

+gradients/huber_loss/Minimum_grad/LessEqual	LessEqualhuber_loss/Abshuber_loss/Minimum/y*
T0*
_output_shapes

:  
с
7gradients/huber_loss/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgs'gradients/huber_loss/Minimum_grad/Shape)gradients/huber_loss/Minimum_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
У
(gradients/huber_loss/Minimum_grad/SelectSelect+gradients/huber_loss/Minimum_grad/LessEqualgradients/AddN_1'gradients/huber_loss/Minimum_grad/zeros*
_output_shapes

:  *
T0
Х
*gradients/huber_loss/Minimum_grad/Select_1Select+gradients/huber_loss/Minimum_grad/LessEqual'gradients/huber_loss/Minimum_grad/zerosgradients/AddN_1*
_output_shapes

:  *
T0
Я
%gradients/huber_loss/Minimum_grad/SumSum(gradients/huber_loss/Minimum_grad/Select7gradients/huber_loss/Minimum_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Л
)gradients/huber_loss/Minimum_grad/ReshapeReshape%gradients/huber_loss/Minimum_grad/Sum'gradients/huber_loss/Minimum_grad/Shape*
T0*
Tshape0*
_output_shapes

:  
е
'gradients/huber_loss/Minimum_grad/Sum_1Sum*gradients/huber_loss/Minimum_grad/Select_19gradients/huber_loss/Minimum_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Й
+gradients/huber_loss/Minimum_grad/Reshape_1Reshape'gradients/huber_loss/Minimum_grad/Sum_1)gradients/huber_loss/Minimum_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 

2gradients/huber_loss/Minimum_grad/tuple/group_depsNoOp*^gradients/huber_loss/Minimum_grad/Reshape,^gradients/huber_loss/Minimum_grad/Reshape_1

:gradients/huber_loss/Minimum_grad/tuple/control_dependencyIdentity)gradients/huber_loss/Minimum_grad/Reshape3^gradients/huber_loss/Minimum_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/huber_loss/Minimum_grad/Reshape*
_output_shapes

:  

<gradients/huber_loss/Minimum_grad/tuple/control_dependency_1Identity+gradients/huber_loss/Minimum_grad/Reshape_13^gradients/huber_loss/Minimum_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/huber_loss/Minimum_grad/Reshape_1*
_output_shapes
: 
ў
gradients/AddN_2AddN8gradients/huber_loss/Sub_1_grad/tuple/control_dependency:gradients/huber_loss/Minimum_grad/tuple/control_dependency*
T0*<
_class2
0.loc:@gradients/huber_loss/Mul_2_grad/Reshape_1*
N*
_output_shapes

:  
c
"gradients/huber_loss/Abs_grad/SignSignhuber_loss/Sub*
_output_shapes

:  *
T0

!gradients/huber_loss/Abs_grad/mulMulgradients/AddN_2"gradients/huber_loss/Abs_grad/Sign*
_output_shapes

:  *
T0
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
е
3gradients/huber_loss/Sub_grad/BroadcastGradientArgsBroadcastGradientArgs#gradients/huber_loss/Sub_grad/Shape%gradients/huber_loss/Sub_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Р
!gradients/huber_loss/Sub_grad/SumSum!gradients/huber_loss/Abs_grad/mul3gradients/huber_loss/Sub_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
И
%gradients/huber_loss/Sub_grad/ReshapeReshape!gradients/huber_loss/Sub_grad/Sum#gradients/huber_loss/Sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
: џџџџџџџџџ
Ф
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
М
'gradients/huber_loss/Sub_grad/Reshape_1Reshape!gradients/huber_loss/Sub_grad/Neg%gradients/huber_loss/Sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ 

.gradients/huber_loss/Sub_grad/tuple/group_depsNoOp&^gradients/huber_loss/Sub_grad/Reshape(^gradients/huber_loss/Sub_grad/Reshape_1

6gradients/huber_loss/Sub_grad/tuple/control_dependencyIdentity%gradients/huber_loss/Sub_grad/Reshape/^gradients/huber_loss/Sub_grad/tuple/group_deps*'
_output_shapes
: џџџџџџџџџ*
T0*8
_class.
,*loc:@gradients/huber_loss/Sub_grad/Reshape

8gradients/huber_loss/Sub_grad/tuple/control_dependency_1Identity'gradients/huber_loss/Sub_grad/Reshape_1/^gradients/huber_loss/Sub_grad/tuple/group_deps*:
_class0
.,loc:@gradients/huber_loss/Sub_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ *
T0
[
gradients/Sum_grad/ShapeShapemul*
T0*
out_type0*
_output_shapes
:

gradients/Sum_grad/SizeConst*+
_class!
loc:@gradients/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 

gradients/Sum_grad/addAddSum/reduction_indicesgradients/Sum_grad/Size*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
: 
Ё
gradients/Sum_grad/modFloorModgradients/Sum_grad/addgradients/Sum_grad/Size*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
: *
T0

gradients/Sum_grad/Shape_1Const*+
_class!
loc:@gradients/Sum_grad/Shape*
valueB *
dtype0*
_output_shapes
: 

gradients/Sum_grad/range/startConst*+
_class!
loc:@gradients/Sum_grad/Shape*
value	B : *
dtype0*
_output_shapes
: 

gradients/Sum_grad/range/deltaConst*+
_class!
loc:@gradients/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
Я
gradients/Sum_grad/rangeRangegradients/Sum_grad/range/startgradients/Sum_grad/Sizegradients/Sum_grad/range/delta*
_output_shapes
:*

Tidx0*+
_class!
loc:@gradients/Sum_grad/Shape

gradients/Sum_grad/Fill/valueConst*+
_class!
loc:@gradients/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
К
gradients/Sum_grad/FillFillgradients/Sum_grad/Shape_1gradients/Sum_grad/Fill/value*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*

index_type0*
_output_shapes
: 
њ
 gradients/Sum_grad/DynamicStitchDynamicStitchgradients/Sum_grad/rangegradients/Sum_grad/modgradients/Sum_grad/Shapegradients/Sum_grad/Fill*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*
N*#
_output_shapes
:џџџџџџџџџ

gradients/Sum_grad/Maximum/yConst*+
_class!
loc:@gradients/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
Р
gradients/Sum_grad/MaximumMaximum gradients/Sum_grad/DynamicStitchgradients/Sum_grad/Maximum/y*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*#
_output_shapes
:џџџџџџџџџ
Џ
gradients/Sum_grad/floordivFloorDivgradients/Sum_grad/Shapegradients/Sum_grad/Maximum*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
:*
T0
А
gradients/Sum_grad/ReshapeReshape6gradients/huber_loss/Sub_grad/tuple/control_dependency gradients/Sum_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
 
gradients/Sum_grad/TileTilegradients/Sum_grad/Reshapegradients/Sum_grad/floordiv*

Tmultiples0*
T0*+
_output_shapes
: џџџџџџџџџ
f
gradients/mul_grad/ShapeShapemain/transpose*
T0*
out_type0*
_output_shapes
:
d
gradients/mul_grad/Shape_1Shape
ExpandDims*
T0*
out_type0*
_output_shapes
:
Д
(gradients/mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_grad/Shapegradients/mul_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
x
gradients/mul_grad/MulMulgradients/Sum_grad/Tile
ExpandDims*
T0*+
_output_shapes
: џџџџџџџџџ

gradients/mul_grad/SumSumgradients/mul_grad/Mul(gradients/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

gradients/mul_grad/ReshapeReshapegradients/mul_grad/Sumgradients/mul_grad/Shape*+
_output_shapes
: џџџџџџџџџ*
T0*
Tshape0
~
gradients/mul_grad/Mul_1Mulmain/transposegradients/Sum_grad/Tile*+
_output_shapes
: џџџџџџџџџ*
T0
Ѕ
gradients/mul_grad/Sum_1Sumgradients/mul_grad/Mul_1*gradients/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ё
gradients/mul_grad/Reshape_1Reshapegradients/mul_grad/Sum_1gradients/mul_grad/Shape_1*
T0*
Tshape0*+
_output_shapes
:џџџџџџџџџ
g
#gradients/mul_grad/tuple/group_depsNoOp^gradients/mul_grad/Reshape^gradients/mul_grad/Reshape_1
о
+gradients/mul_grad/tuple/control_dependencyIdentitygradients/mul_grad/Reshape$^gradients/mul_grad/tuple/group_deps*-
_class#
!loc:@gradients/mul_grad/Reshape*+
_output_shapes
: џџџџџџџџџ*
T0
ф
-gradients/mul_grad/tuple/control_dependency_1Identitygradients/mul_grad/Reshape_1$^gradients/mul_grad/tuple/group_deps*+
_output_shapes
:џџџџџџџџџ*
T0*/
_class%
#!loc:@gradients/mul_grad/Reshape_1
~
/gradients/main/transpose_grad/InvertPermutationInvertPermutationmain/transpose/perm*
T0*
_output_shapes
:
е
'gradients/main/transpose_grad/transpose	Transpose+gradients/mul_grad/tuple/control_dependency/gradients/main/transpose_grad/InvertPermutation*+
_output_shapes
: џџџџџџџџџ*
Tperm0*
T0
ѓ
'gradients/main/transpose/x_grad/unstackUnpack'gradients/main/transpose_grad/transpose*	
num *
T0*

axis *і
_output_shapesу
р:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ
b
0gradients/main/transpose/x_grad/tuple/group_depsNoOp(^gradients/main/transpose/x_grad/unstack

8gradients/main/transpose/x_grad/tuple/control_dependencyIdentity'gradients/main/transpose/x_grad/unstack1^gradients/main/transpose/x_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack

:gradients/main/transpose/x_grad/tuple/control_dependency_1Identity)gradients/main/transpose/x_grad/unstack:11^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:џџџџџџџџџ

:gradients/main/transpose/x_grad/tuple/control_dependency_2Identity)gradients/main/transpose/x_grad/unstack:21^gradients/main/transpose/x_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack

:gradients/main/transpose/x_grad/tuple/control_dependency_3Identity)gradients/main/transpose/x_grad/unstack:31^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:џџџџџџџџџ

:gradients/main/transpose/x_grad/tuple/control_dependency_4Identity)gradients/main/transpose/x_grad/unstack:41^gradients/main/transpose/x_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack

:gradients/main/transpose/x_grad/tuple/control_dependency_5Identity)gradients/main/transpose/x_grad/unstack:51^gradients/main/transpose/x_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack

:gradients/main/transpose/x_grad/tuple/control_dependency_6Identity)gradients/main/transpose/x_grad/unstack:61^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:џџџџџџџџџ

:gradients/main/transpose/x_grad/tuple/control_dependency_7Identity)gradients/main/transpose/x_grad/unstack:71^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:џџџџџџџџџ

:gradients/main/transpose/x_grad/tuple/control_dependency_8Identity)gradients/main/transpose/x_grad/unstack:81^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:џџџџџџџџџ

:gradients/main/transpose/x_grad/tuple/control_dependency_9Identity)gradients/main/transpose/x_grad/unstack:91^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:џџџџџџџџџ

;gradients/main/transpose/x_grad/tuple/control_dependency_10Identity*gradients/main/transpose/x_grad/unstack:101^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:џџџџџџџџџ

;gradients/main/transpose/x_grad/tuple/control_dependency_11Identity*gradients/main/transpose/x_grad/unstack:111^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:џџџџџџџџџ

;gradients/main/transpose/x_grad/tuple/control_dependency_12Identity*gradients/main/transpose/x_grad/unstack:121^gradients/main/transpose/x_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack

;gradients/main/transpose/x_grad/tuple/control_dependency_13Identity*gradients/main/transpose/x_grad/unstack:131^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:џџџџџџџџџ

;gradients/main/transpose/x_grad/tuple/control_dependency_14Identity*gradients/main/transpose/x_grad/unstack:141^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:џџџџџџџџџ

;gradients/main/transpose/x_grad/tuple/control_dependency_15Identity*gradients/main/transpose/x_grad/unstack:151^gradients/main/transpose/x_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack

;gradients/main/transpose/x_grad/tuple/control_dependency_16Identity*gradients/main/transpose/x_grad/unstack:161^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:џџџџџџџџџ

;gradients/main/transpose/x_grad/tuple/control_dependency_17Identity*gradients/main/transpose/x_grad/unstack:171^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:џџџџџџџџџ

;gradients/main/transpose/x_grad/tuple/control_dependency_18Identity*gradients/main/transpose/x_grad/unstack:181^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:џџџџџџџџџ

;gradients/main/transpose/x_grad/tuple/control_dependency_19Identity*gradients/main/transpose/x_grad/unstack:191^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:џџџџџџџџџ

;gradients/main/transpose/x_grad/tuple/control_dependency_20Identity*gradients/main/transpose/x_grad/unstack:201^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:џџџџџџџџџ

;gradients/main/transpose/x_grad/tuple/control_dependency_21Identity*gradients/main/transpose/x_grad/unstack:211^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:џџџџџџџџџ

;gradients/main/transpose/x_grad/tuple/control_dependency_22Identity*gradients/main/transpose/x_grad/unstack:221^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:џџџџџџџџџ

;gradients/main/transpose/x_grad/tuple/control_dependency_23Identity*gradients/main/transpose/x_grad/unstack:231^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:џџџџџџџџџ

;gradients/main/transpose/x_grad/tuple/control_dependency_24Identity*gradients/main/transpose/x_grad/unstack:241^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:џџџџџџџџџ

;gradients/main/transpose/x_grad/tuple/control_dependency_25Identity*gradients/main/transpose/x_grad/unstack:251^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:џџџџџџџџџ

;gradients/main/transpose/x_grad/tuple/control_dependency_26Identity*gradients/main/transpose/x_grad/unstack:261^gradients/main/transpose/x_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack

;gradients/main/transpose/x_grad/tuple/control_dependency_27Identity*gradients/main/transpose/x_grad/unstack:271^gradients/main/transpose/x_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack

;gradients/main/transpose/x_grad/tuple/control_dependency_28Identity*gradients/main/transpose/x_grad/unstack:281^gradients/main/transpose/x_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack

;gradients/main/transpose/x_grad/tuple/control_dependency_29Identity*gradients/main/transpose/x_grad/unstack:291^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:џџџџџџџџџ

;gradients/main/transpose/x_grad/tuple/control_dependency_30Identity*gradients/main/transpose/x_grad/unstack:301^gradients/main/transpose/x_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack

;gradients/main/transpose/x_grad/tuple/control_dependency_31Identity*gradients/main/transpose/x_grad/unstack:311^gradients/main/transpose/x_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack

 gradients/main/split_grad/concatConcatV28gradients/main/transpose/x_grad/tuple/control_dependency:gradients/main/transpose/x_grad/tuple/control_dependency_1:gradients/main/transpose/x_grad/tuple/control_dependency_2:gradients/main/transpose/x_grad/tuple/control_dependency_3:gradients/main/transpose/x_grad/tuple/control_dependency_4:gradients/main/transpose/x_grad/tuple/control_dependency_5:gradients/main/transpose/x_grad/tuple/control_dependency_6:gradients/main/transpose/x_grad/tuple/control_dependency_7:gradients/main/transpose/x_grad/tuple/control_dependency_8:gradients/main/transpose/x_grad/tuple/control_dependency_9;gradients/main/transpose/x_grad/tuple/control_dependency_10;gradients/main/transpose/x_grad/tuple/control_dependency_11;gradients/main/transpose/x_grad/tuple/control_dependency_12;gradients/main/transpose/x_grad/tuple/control_dependency_13;gradients/main/transpose/x_grad/tuple/control_dependency_14;gradients/main/transpose/x_grad/tuple/control_dependency_15;gradients/main/transpose/x_grad/tuple/control_dependency_16;gradients/main/transpose/x_grad/tuple/control_dependency_17;gradients/main/transpose/x_grad/tuple/control_dependency_18;gradients/main/transpose/x_grad/tuple/control_dependency_19;gradients/main/transpose/x_grad/tuple/control_dependency_20;gradients/main/transpose/x_grad/tuple/control_dependency_21;gradients/main/transpose/x_grad/tuple/control_dependency_22;gradients/main/transpose/x_grad/tuple/control_dependency_23;gradients/main/transpose/x_grad/tuple/control_dependency_24;gradients/main/transpose/x_grad/tuple/control_dependency_25;gradients/main/transpose/x_grad/tuple/control_dependency_26;gradients/main/transpose/x_grad/tuple/control_dependency_27;gradients/main/transpose/x_grad/tuple/control_dependency_28;gradients/main/transpose/x_grad/tuple/control_dependency_29;gradients/main/transpose/x_grad/tuple/control_dependency_30;gradients/main/transpose/x_grad/tuple/control_dependency_31main/split/split_dim*
T0*
N *'
_output_shapes
:џџџџџџџџџ*

Tidx0

/gradients/main/dense_4/BiasAdd_grad/BiasAddGradBiasAddGrad gradients/main/split_grad/concat*
_output_shapes
:*
T0*
data_formatNHWC

4gradients/main/dense_4/BiasAdd_grad/tuple/group_depsNoOp0^gradients/main/dense_4/BiasAdd_grad/BiasAddGrad!^gradients/main/split_grad/concat

<gradients/main/dense_4/BiasAdd_grad/tuple/control_dependencyIdentity gradients/main/split_grad/concat5^gradients/main/dense_4/BiasAdd_grad/tuple/group_deps*3
_class)
'%loc:@gradients/main/split_grad/concat*'
_output_shapes
:џџџџџџџџџ*
T0

>gradients/main/dense_4/BiasAdd_grad/tuple/control_dependency_1Identity/gradients/main/dense_4/BiasAdd_grad/BiasAddGrad5^gradients/main/dense_4/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*B
_class8
64loc:@gradients/main/dense_4/BiasAdd_grad/BiasAddGrad
ф
)gradients/main/dense_4/MatMul_grad/MatMulMatMul<gradients/main/dense_4/BiasAdd_grad/tuple/control_dependencymain/dense_4/kernel/read*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(*
T0
ж
+gradients/main/dense_4/MatMul_grad/MatMul_1MatMulmain/dense_3/Relu<gradients/main/dense_4/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	*
transpose_a(*
transpose_b( *
T0

3gradients/main/dense_4/MatMul_grad/tuple/group_depsNoOp*^gradients/main/dense_4/MatMul_grad/MatMul,^gradients/main/dense_4/MatMul_grad/MatMul_1

;gradients/main/dense_4/MatMul_grad/tuple/control_dependencyIdentity)gradients/main/dense_4/MatMul_grad/MatMul4^gradients/main/dense_4/MatMul_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/main/dense_4/MatMul_grad/MatMul*(
_output_shapes
:џџџџџџџџџ

=gradients/main/dense_4/MatMul_grad/tuple/control_dependency_1Identity+gradients/main/dense_4/MatMul_grad/MatMul_14^gradients/main/dense_4/MatMul_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/main/dense_4/MatMul_grad/MatMul_1*
_output_shapes
:	
И
)gradients/main/dense_3/Relu_grad/ReluGradReluGrad;gradients/main/dense_4/MatMul_grad/tuple/control_dependencymain/dense_3/Relu*
T0*(
_output_shapes
:џџџџџџџџџ
І
/gradients/main/dense_3/BiasAdd_grad/BiasAddGradBiasAddGrad)gradients/main/dense_3/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:

4gradients/main/dense_3/BiasAdd_grad/tuple/group_depsNoOp0^gradients/main/dense_3/BiasAdd_grad/BiasAddGrad*^gradients/main/dense_3/Relu_grad/ReluGrad

<gradients/main/dense_3/BiasAdd_grad/tuple/control_dependencyIdentity)gradients/main/dense_3/Relu_grad/ReluGrad5^gradients/main/dense_3/BiasAdd_grad/tuple/group_deps*<
_class2
0.loc:@gradients/main/dense_3/Relu_grad/ReluGrad*(
_output_shapes
:џџџџџџџџџ*
T0

>gradients/main/dense_3/BiasAdd_grad/tuple/control_dependency_1Identity/gradients/main/dense_3/BiasAdd_grad/BiasAddGrad5^gradients/main/dense_3/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:*
T0*B
_class8
64loc:@gradients/main/dense_3/BiasAdd_grad/BiasAddGrad
ф
)gradients/main/dense_3/MatMul_grad/MatMulMatMul<gradients/main/dense_3/BiasAdd_grad/tuple/control_dependencymain/dense_3/kernel/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(
з
+gradients/main/dense_3/MatMul_grad/MatMul_1MatMulmain/dense_2/Relu<gradients/main/dense_3/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
*
transpose_a(*
transpose_b( *
T0

3gradients/main/dense_3/MatMul_grad/tuple/group_depsNoOp*^gradients/main/dense_3/MatMul_grad/MatMul,^gradients/main/dense_3/MatMul_grad/MatMul_1

;gradients/main/dense_3/MatMul_grad/tuple/control_dependencyIdentity)gradients/main/dense_3/MatMul_grad/MatMul4^gradients/main/dense_3/MatMul_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*<
_class2
0.loc:@gradients/main/dense_3/MatMul_grad/MatMul

=gradients/main/dense_3/MatMul_grad/tuple/control_dependency_1Identity+gradients/main/dense_3/MatMul_grad/MatMul_14^gradients/main/dense_3/MatMul_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/main/dense_3/MatMul_grad/MatMul_1* 
_output_shapes
:

И
)gradients/main/dense_2/Relu_grad/ReluGradReluGrad;gradients/main/dense_3/MatMul_grad/tuple/control_dependencymain/dense_2/Relu*
T0*(
_output_shapes
:џџџџџџџџџ
І
/gradients/main/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad)gradients/main/dense_2/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes	
:*
T0

4gradients/main/dense_2/BiasAdd_grad/tuple/group_depsNoOp0^gradients/main/dense_2/BiasAdd_grad/BiasAddGrad*^gradients/main/dense_2/Relu_grad/ReluGrad

<gradients/main/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity)gradients/main/dense_2/Relu_grad/ReluGrad5^gradients/main/dense_2/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*<
_class2
0.loc:@gradients/main/dense_2/Relu_grad/ReluGrad

>gradients/main/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity/gradients/main/dense_2/BiasAdd_grad/BiasAddGrad5^gradients/main/dense_2/BiasAdd_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/main/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
ф
)gradients/main/dense_2/MatMul_grad/MatMulMatMul<gradients/main/dense_2/BiasAdd_grad/tuple/control_dependencymain/dense_2/kernel/read*
transpose_b(*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( 
Ю
+gradients/main/dense_2/MatMul_grad/MatMul_1MatMulmain/Mul<gradients/main/dense_2/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
*
transpose_a(*
transpose_b( 

3gradients/main/dense_2/MatMul_grad/tuple/group_depsNoOp*^gradients/main/dense_2/MatMul_grad/MatMul,^gradients/main/dense_2/MatMul_grad/MatMul_1

;gradients/main/dense_2/MatMul_grad/tuple/control_dependencyIdentity)gradients/main/dense_2/MatMul_grad/MatMul4^gradients/main/dense_2/MatMul_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/main/dense_2/MatMul_grad/MatMul*(
_output_shapes
:џџџџџџџџџ

=gradients/main/dense_2/MatMul_grad/tuple/control_dependency_1Identity+gradients/main/dense_2/MatMul_grad/MatMul_14^gradients/main/dense_2/MatMul_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/main/dense_2/MatMul_grad/MatMul_1* 
_output_shapes
:

l
gradients/main/Mul_grad/ShapeShapemain/dense/Selu*
_output_shapes
:*
T0*
out_type0
p
gradients/main/Mul_grad/Shape_1Shapemain/dense_1/Relu*
T0*
out_type0*
_output_shapes
:
У
-gradients/main/Mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/main/Mul_grad/Shapegradients/main/Mul_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Ѕ
gradients/main/Mul_grad/MulMul;gradients/main/dense_2/MatMul_grad/tuple/control_dependencymain/dense_1/Relu*
T0*(
_output_shapes
:џџџџџџџџџ
Ў
gradients/main/Mul_grad/SumSumgradients/main/Mul_grad/Mul-gradients/main/Mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ї
gradients/main/Mul_grad/ReshapeReshapegradients/main/Mul_grad/Sumgradients/main/Mul_grad/Shape*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
Ѕ
gradients/main/Mul_grad/Mul_1Mulmain/dense/Selu;gradients/main/dense_2/MatMul_grad/tuple/control_dependency*(
_output_shapes
:џџџџџџџџџ*
T0
Д
gradients/main/Mul_grad/Sum_1Sumgradients/main/Mul_grad/Mul_1/gradients/main/Mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
­
!gradients/main/Mul_grad/Reshape_1Reshapegradients/main/Mul_grad/Sum_1gradients/main/Mul_grad/Shape_1*
Tshape0*(
_output_shapes
:џџџџџџџџџ*
T0
v
(gradients/main/Mul_grad/tuple/group_depsNoOp ^gradients/main/Mul_grad/Reshape"^gradients/main/Mul_grad/Reshape_1
я
0gradients/main/Mul_grad/tuple/control_dependencyIdentitygradients/main/Mul_grad/Reshape)^gradients/main/Mul_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/main/Mul_grad/Reshape*(
_output_shapes
:џџџџџџџџџ
ѕ
2gradients/main/Mul_grad/tuple/control_dependency_1Identity!gradients/main/Mul_grad/Reshape_1)^gradients/main/Mul_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*4
_class*
(&loc:@gradients/main/Mul_grad/Reshape_1
Љ
'gradients/main/dense/Selu_grad/SeluGradSeluGrad0gradients/main/Mul_grad/tuple/control_dependencymain/dense/Selu*
T0*(
_output_shapes
:џџџџџџџџџ
Џ
)gradients/main/dense_1/Relu_grad/ReluGradReluGrad2gradients/main/Mul_grad/tuple/control_dependency_1main/dense_1/Relu*(
_output_shapes
:џџџџџџџџџ*
T0
Ђ
-gradients/main/dense/BiasAdd_grad/BiasAddGradBiasAddGrad'gradients/main/dense/Selu_grad/SeluGrad*
T0*
data_formatNHWC*
_output_shapes	
:

2gradients/main/dense/BiasAdd_grad/tuple/group_depsNoOp.^gradients/main/dense/BiasAdd_grad/BiasAddGrad(^gradients/main/dense/Selu_grad/SeluGrad

:gradients/main/dense/BiasAdd_grad/tuple/control_dependencyIdentity'gradients/main/dense/Selu_grad/SeluGrad3^gradients/main/dense/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*:
_class0
.,loc:@gradients/main/dense/Selu_grad/SeluGrad

<gradients/main/dense/BiasAdd_grad/tuple/control_dependency_1Identity-gradients/main/dense/BiasAdd_grad/BiasAddGrad3^gradients/main/dense/BiasAdd_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/main/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
І
/gradients/main/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad)gradients/main/dense_1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:

4gradients/main/dense_1/BiasAdd_grad/tuple/group_depsNoOp0^gradients/main/dense_1/BiasAdd_grad/BiasAddGrad*^gradients/main/dense_1/Relu_grad/ReluGrad

<gradients/main/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity)gradients/main/dense_1/Relu_grad/ReluGrad5^gradients/main/dense_1/BiasAdd_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/main/dense_1/Relu_grad/ReluGrad*(
_output_shapes
:џџџџџџџџџ

>gradients/main/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity/gradients/main/dense_1/BiasAdd_grad/BiasAddGrad5^gradients/main/dense_1/BiasAdd_grad/tuple/group_deps*B
_class8
64loc:@gradients/main/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
н
'gradients/main/dense/MatMul_grad/MatMulMatMul:gradients/main/dense/BiasAdd_grad/tuple/control_dependencymain/dense/kernel/read*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(*
T0
Э
)gradients/main/dense/MatMul_grad/MatMul_1MatMulmain/Reshape:gradients/main/dense/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	*
transpose_a(*
transpose_b( *
T0

1gradients/main/dense/MatMul_grad/tuple/group_depsNoOp(^gradients/main/dense/MatMul_grad/MatMul*^gradients/main/dense/MatMul_grad/MatMul_1

9gradients/main/dense/MatMul_grad/tuple/control_dependencyIdentity'gradients/main/dense/MatMul_grad/MatMul2^gradients/main/dense/MatMul_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/dense/MatMul_grad/MatMul*'
_output_shapes
:џџџџџџџџџ

;gradients/main/dense/MatMul_grad/tuple/control_dependency_1Identity)gradients/main/dense/MatMul_grad/MatMul_12^gradients/main/dense/MatMul_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/main/dense/MatMul_grad/MatMul_1*
_output_shapes
:	
у
)gradients/main/dense_1/MatMul_grad/MatMulMatMul<gradients/main/dense_1/BiasAdd_grad/tuple/control_dependencymain/dense_1/kernel/read*
T0*'
_output_shapes
:џџџџџџџџџ@*
transpose_a( *
transpose_b(
Э
+gradients/main/dense_1/MatMul_grad/MatMul_1MatMulmain/Cos<gradients/main/dense_1/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	@*
transpose_a(*
transpose_b( *
T0

3gradients/main/dense_1/MatMul_grad/tuple/group_depsNoOp*^gradients/main/dense_1/MatMul_grad/MatMul,^gradients/main/dense_1/MatMul_grad/MatMul_1

;gradients/main/dense_1/MatMul_grad/tuple/control_dependencyIdentity)gradients/main/dense_1/MatMul_grad/MatMul4^gradients/main/dense_1/MatMul_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ@*
T0*<
_class2
0.loc:@gradients/main/dense_1/MatMul_grad/MatMul

=gradients/main/dense_1/MatMul_grad/tuple/control_dependency_1Identity+gradients/main/dense_1/MatMul_grad/MatMul_14^gradients/main/dense_1/MatMul_grad/tuple/group_deps*
_output_shapes
:	@*
T0*>
_class4
20loc:@gradients/main/dense_1/MatMul_grad/MatMul_1

beta1_power/initial_valueConst*"
_class
loc:@main/dense/bias*
valueB
 *fff?*
dtype0*
_output_shapes
: 

beta1_power
VariableV2*"
_class
loc:@main/dense/bias*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
В
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*"
_class
loc:@main/dense/bias
n
beta1_power/readIdentitybeta1_power*"
_class
loc:@main/dense/bias*
_output_shapes
: *
T0

beta2_power/initial_valueConst*"
_class
loc:@main/dense/bias*
valueB
 *wО?*
dtype0*
_output_shapes
: 

beta2_power
VariableV2*
_output_shapes
: *
shared_name *"
_class
loc:@main/dense/bias*
	container *
shape: *
dtype0
В
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
Ѕ
(main/dense/kernel/Adam/Initializer/zerosConst*
_output_shapes
:	*$
_class
loc:@main/dense/kernel*
valueB	*    *
dtype0
В
main/dense/kernel/Adam
VariableV2*
dtype0*
_output_shapes
:	*
shared_name *$
_class
loc:@main/dense/kernel*
	container *
shape:	
т
main/dense/kernel/Adam/AssignAssignmain/dense/kernel/Adam(main/dense/kernel/Adam/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@main/dense/kernel*
validate_shape(*
_output_shapes
:	

main/dense/kernel/Adam/readIdentitymain/dense/kernel/Adam*$
_class
loc:@main/dense/kernel*
_output_shapes
:	*
T0
Ї
*main/dense/kernel/Adam_1/Initializer/zerosConst*$
_class
loc:@main/dense/kernel*
valueB	*    *
dtype0*
_output_shapes
:	
Д
main/dense/kernel/Adam_1
VariableV2*
dtype0*
_output_shapes
:	*
shared_name *$
_class
loc:@main/dense/kernel*
	container *
shape:	
ш
main/dense/kernel/Adam_1/AssignAssignmain/dense/kernel/Adam_1*main/dense/kernel/Adam_1/Initializer/zeros*$
_class
loc:@main/dense/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0

main/dense/kernel/Adam_1/readIdentitymain/dense/kernel/Adam_1*$
_class
loc:@main/dense/kernel*
_output_shapes
:	*
T0

&main/dense/bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes	
:*"
_class
loc:@main/dense/bias*
valueB*    
І
main/dense/bias/Adam
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *"
_class
loc:@main/dense/bias*
	container *
shape:
ж
main/dense/bias/Adam/AssignAssignmain/dense/bias/Adam&main/dense/bias/Adam/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*"
_class
loc:@main/dense/bias*
validate_shape(

main/dense/bias/Adam/readIdentitymain/dense/bias/Adam*"
_class
loc:@main/dense/bias*
_output_shapes	
:*
T0

(main/dense/bias/Adam_1/Initializer/zerosConst*"
_class
loc:@main/dense/bias*
valueB*    *
dtype0*
_output_shapes	
:
Ј
main/dense/bias/Adam_1
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *"
_class
loc:@main/dense/bias
м
main/dense/bias/Adam_1/AssignAssignmain/dense/bias/Adam_1(main/dense/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@main/dense/bias*
validate_shape(*
_output_shapes	
:

main/dense/bias/Adam_1/readIdentitymain/dense/bias/Adam_1*
T0*"
_class
loc:@main/dense/bias*
_output_shapes	
:
Г
:main/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*&
_class
loc:@main/dense_1/kernel*
valueB"@      *
dtype0*
_output_shapes
:

0main/dense_1/kernel/Adam/Initializer/zeros/ConstConst*&
_class
loc:@main/dense_1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

*main/dense_1/kernel/Adam/Initializer/zerosFill:main/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensor0main/dense_1/kernel/Adam/Initializer/zeros/Const*&
_class
loc:@main/dense_1/kernel*

index_type0*
_output_shapes
:	@*
T0
Ж
main/dense_1/kernel/Adam
VariableV2*
shared_name *&
_class
loc:@main/dense_1/kernel*
	container *
shape:	@*
dtype0*
_output_shapes
:	@
ъ
main/dense_1/kernel/Adam/AssignAssignmain/dense_1/kernel/Adam*main/dense_1/kernel/Adam/Initializer/zeros*
use_locking(*
T0*&
_class
loc:@main/dense_1/kernel*
validate_shape(*
_output_shapes
:	@

main/dense_1/kernel/Adam/readIdentitymain/dense_1/kernel/Adam*
_output_shapes
:	@*
T0*&
_class
loc:@main/dense_1/kernel
Е
<main/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*&
_class
loc:@main/dense_1/kernel*
valueB"@      *
dtype0*
_output_shapes
:

2main/dense_1/kernel/Adam_1/Initializer/zeros/ConstConst*&
_class
loc:@main/dense_1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

,main/dense_1/kernel/Adam_1/Initializer/zerosFill<main/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor2main/dense_1/kernel/Adam_1/Initializer/zeros/Const*
_output_shapes
:	@*
T0*&
_class
loc:@main/dense_1/kernel*

index_type0
И
main/dense_1/kernel/Adam_1
VariableV2*
shared_name *&
_class
loc:@main/dense_1/kernel*
	container *
shape:	@*
dtype0*
_output_shapes
:	@
№
!main/dense_1/kernel/Adam_1/AssignAssignmain/dense_1/kernel/Adam_1,main/dense_1/kernel/Adam_1/Initializer/zeros*
_output_shapes
:	@*
use_locking(*
T0*&
_class
loc:@main/dense_1/kernel*
validate_shape(

main/dense_1/kernel/Adam_1/readIdentitymain/dense_1/kernel/Adam_1*
T0*&
_class
loc:@main/dense_1/kernel*
_output_shapes
:	@

(main/dense_1/bias/Adam/Initializer/zerosConst*
_output_shapes	
:*$
_class
loc:@main/dense_1/bias*
valueB*    *
dtype0
Њ
main/dense_1/bias/Adam
VariableV2*
shared_name *$
_class
loc:@main/dense_1/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
о
main/dense_1/bias/Adam/AssignAssignmain/dense_1/bias/Adam(main/dense_1/bias/Adam/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*$
_class
loc:@main/dense_1/bias*
validate_shape(

main/dense_1/bias/Adam/readIdentitymain/dense_1/bias/Adam*
T0*$
_class
loc:@main/dense_1/bias*
_output_shapes	
:

*main/dense_1/bias/Adam_1/Initializer/zerosConst*$
_class
loc:@main/dense_1/bias*
valueB*    *
dtype0*
_output_shapes	
:
Ќ
main/dense_1/bias/Adam_1
VariableV2*
_output_shapes	
:*
shared_name *$
_class
loc:@main/dense_1/bias*
	container *
shape:*
dtype0
ф
main/dense_1/bias/Adam_1/AssignAssignmain/dense_1/bias/Adam_1*main/dense_1/bias/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*$
_class
loc:@main/dense_1/bias

main/dense_1/bias/Adam_1/readIdentitymain/dense_1/bias/Adam_1*
_output_shapes	
:*
T0*$
_class
loc:@main/dense_1/bias
Г
:main/dense_2/kernel/Adam/Initializer/zeros/shape_as_tensorConst*&
_class
loc:@main/dense_2/kernel*
valueB"      *
dtype0*
_output_shapes
:

0main/dense_2/kernel/Adam/Initializer/zeros/ConstConst*&
_class
loc:@main/dense_2/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

*main/dense_2/kernel/Adam/Initializer/zerosFill:main/dense_2/kernel/Adam/Initializer/zeros/shape_as_tensor0main/dense_2/kernel/Adam/Initializer/zeros/Const*
T0*&
_class
loc:@main/dense_2/kernel*

index_type0* 
_output_shapes
:

И
main/dense_2/kernel/Adam
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *&
_class
loc:@main/dense_2/kernel*
	container *
shape:

ы
main/dense_2/kernel/Adam/AssignAssignmain/dense_2/kernel/Adam*main/dense_2/kernel/Adam/Initializer/zeros*
use_locking(*
T0*&
_class
loc:@main/dense_2/kernel*
validate_shape(* 
_output_shapes
:


main/dense_2/kernel/Adam/readIdentitymain/dense_2/kernel/Adam*
T0*&
_class
loc:@main/dense_2/kernel* 
_output_shapes
:

Е
<main/dense_2/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*&
_class
loc:@main/dense_2/kernel*
valueB"      *
dtype0*
_output_shapes
:

2main/dense_2/kernel/Adam_1/Initializer/zeros/ConstConst*&
_class
loc:@main/dense_2/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

,main/dense_2/kernel/Adam_1/Initializer/zerosFill<main/dense_2/kernel/Adam_1/Initializer/zeros/shape_as_tensor2main/dense_2/kernel/Adam_1/Initializer/zeros/Const* 
_output_shapes
:
*
T0*&
_class
loc:@main/dense_2/kernel*

index_type0
К
main/dense_2/kernel/Adam_1
VariableV2*
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *&
_class
loc:@main/dense_2/kernel*
	container 
ё
!main/dense_2/kernel/Adam_1/AssignAssignmain/dense_2/kernel/Adam_1,main/dense_2/kernel/Adam_1/Initializer/zeros*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*&
_class
loc:@main/dense_2/kernel

main/dense_2/kernel/Adam_1/readIdentitymain/dense_2/kernel/Adam_1*
T0*&
_class
loc:@main/dense_2/kernel* 
_output_shapes
:


(main/dense_2/bias/Adam/Initializer/zerosConst*
_output_shapes	
:*$
_class
loc:@main/dense_2/bias*
valueB*    *
dtype0
Њ
main/dense_2/bias/Adam
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *$
_class
loc:@main/dense_2/bias*
	container *
shape:
о
main/dense_2/bias/Adam/AssignAssignmain/dense_2/bias/Adam(main/dense_2/bias/Adam/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*$
_class
loc:@main/dense_2/bias*
validate_shape(

main/dense_2/bias/Adam/readIdentitymain/dense_2/bias/Adam*
T0*$
_class
loc:@main/dense_2/bias*
_output_shapes	
:

*main/dense_2/bias/Adam_1/Initializer/zerosConst*$
_class
loc:@main/dense_2/bias*
valueB*    *
dtype0*
_output_shapes	
:
Ќ
main/dense_2/bias/Adam_1
VariableV2*
_output_shapes	
:*
shared_name *$
_class
loc:@main/dense_2/bias*
	container *
shape:*
dtype0
ф
main/dense_2/bias/Adam_1/AssignAssignmain/dense_2/bias/Adam_1*main/dense_2/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@main/dense_2/bias*
validate_shape(*
_output_shapes	
:

main/dense_2/bias/Adam_1/readIdentitymain/dense_2/bias/Adam_1*$
_class
loc:@main/dense_2/bias*
_output_shapes	
:*
T0
Г
:main/dense_3/kernel/Adam/Initializer/zeros/shape_as_tensorConst*&
_class
loc:@main/dense_3/kernel*
valueB"      *
dtype0*
_output_shapes
:

0main/dense_3/kernel/Adam/Initializer/zeros/ConstConst*&
_class
loc:@main/dense_3/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

*main/dense_3/kernel/Adam/Initializer/zerosFill:main/dense_3/kernel/Adam/Initializer/zeros/shape_as_tensor0main/dense_3/kernel/Adam/Initializer/zeros/Const*
T0*&
_class
loc:@main/dense_3/kernel*

index_type0* 
_output_shapes
:

И
main/dense_3/kernel/Adam
VariableV2*
shared_name *&
_class
loc:@main/dense_3/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:

ы
main/dense_3/kernel/Adam/AssignAssignmain/dense_3/kernel/Adam*main/dense_3/kernel/Adam/Initializer/zeros*
T0*&
_class
loc:@main/dense_3/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(

main/dense_3/kernel/Adam/readIdentitymain/dense_3/kernel/Adam* 
_output_shapes
:
*
T0*&
_class
loc:@main/dense_3/kernel
Е
<main/dense_3/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*&
_class
loc:@main/dense_3/kernel*
valueB"      *
dtype0*
_output_shapes
:

2main/dense_3/kernel/Adam_1/Initializer/zeros/ConstConst*&
_class
loc:@main/dense_3/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

,main/dense_3/kernel/Adam_1/Initializer/zerosFill<main/dense_3/kernel/Adam_1/Initializer/zeros/shape_as_tensor2main/dense_3/kernel/Adam_1/Initializer/zeros/Const*
T0*&
_class
loc:@main/dense_3/kernel*

index_type0* 
_output_shapes
:

К
main/dense_3/kernel/Adam_1
VariableV2*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *&
_class
loc:@main/dense_3/kernel
ё
!main/dense_3/kernel/Adam_1/AssignAssignmain/dense_3/kernel/Adam_1,main/dense_3/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*&
_class
loc:@main/dense_3/kernel*
validate_shape(* 
_output_shapes
:


main/dense_3/kernel/Adam_1/readIdentitymain/dense_3/kernel/Adam_1*
T0*&
_class
loc:@main/dense_3/kernel* 
_output_shapes
:


(main/dense_3/bias/Adam/Initializer/zerosConst*$
_class
loc:@main/dense_3/bias*
valueB*    *
dtype0*
_output_shapes	
:
Њ
main/dense_3/bias/Adam
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *$
_class
loc:@main/dense_3/bias
о
main/dense_3/bias/Adam/AssignAssignmain/dense_3/bias/Adam(main/dense_3/bias/Adam/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*$
_class
loc:@main/dense_3/bias*
validate_shape(

main/dense_3/bias/Adam/readIdentitymain/dense_3/bias/Adam*
_output_shapes	
:*
T0*$
_class
loc:@main/dense_3/bias

*main/dense_3/bias/Adam_1/Initializer/zerosConst*$
_class
loc:@main/dense_3/bias*
valueB*    *
dtype0*
_output_shapes	
:
Ќ
main/dense_3/bias/Adam_1
VariableV2*$
_class
loc:@main/dense_3/bias*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
ф
main/dense_3/bias/Adam_1/AssignAssignmain/dense_3/bias/Adam_1*main/dense_3/bias/Adam_1/Initializer/zeros*
T0*$
_class
loc:@main/dense_3/bias*
validate_shape(*
_output_shapes	
:*
use_locking(

main/dense_3/bias/Adam_1/readIdentitymain/dense_3/bias/Adam_1*
T0*$
_class
loc:@main/dense_3/bias*
_output_shapes	
:
Љ
*main/dense_4/kernel/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:	*&
_class
loc:@main/dense_4/kernel*
valueB	*    
Ж
main/dense_4/kernel/Adam
VariableV2*
shared_name *&
_class
loc:@main/dense_4/kernel*
	container *
shape:	*
dtype0*
_output_shapes
:	
ъ
main/dense_4/kernel/Adam/AssignAssignmain/dense_4/kernel/Adam*main/dense_4/kernel/Adam/Initializer/zeros*
use_locking(*
T0*&
_class
loc:@main/dense_4/kernel*
validate_shape(*
_output_shapes
:	

main/dense_4/kernel/Adam/readIdentitymain/dense_4/kernel/Adam*
T0*&
_class
loc:@main/dense_4/kernel*
_output_shapes
:	
Ћ
,main/dense_4/kernel/Adam_1/Initializer/zerosConst*&
_class
loc:@main/dense_4/kernel*
valueB	*    *
dtype0*
_output_shapes
:	
И
main/dense_4/kernel/Adam_1
VariableV2*
shape:	*
dtype0*
_output_shapes
:	*
shared_name *&
_class
loc:@main/dense_4/kernel*
	container 
№
!main/dense_4/kernel/Adam_1/AssignAssignmain/dense_4/kernel/Adam_1,main/dense_4/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*&
_class
loc:@main/dense_4/kernel*
validate_shape(*
_output_shapes
:	

main/dense_4/kernel/Adam_1/readIdentitymain/dense_4/kernel/Adam_1*
T0*&
_class
loc:@main/dense_4/kernel*
_output_shapes
:	

(main/dense_4/bias/Adam/Initializer/zerosConst*$
_class
loc:@main/dense_4/bias*
valueB*    *
dtype0*
_output_shapes
:
Ј
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
н
main/dense_4/bias/Adam/AssignAssignmain/dense_4/bias/Adam(main/dense_4/bias/Adam/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@main/dense_4/bias*
validate_shape(*
_output_shapes
:

main/dense_4/bias/Adam/readIdentitymain/dense_4/bias/Adam*
_output_shapes
:*
T0*$
_class
loc:@main/dense_4/bias

*main/dense_4/bias/Adam_1/Initializer/zerosConst*$
_class
loc:@main/dense_4/bias*
valueB*    *
dtype0*
_output_shapes
:
Њ
main/dense_4/bias/Adam_1
VariableV2*
shared_name *$
_class
loc:@main/dense_4/bias*
	container *
shape:*
dtype0*
_output_shapes
:
у
main/dense_4/bias/Adam_1/AssignAssignmain/dense_4/bias/Adam_1*main/dense_4/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@main/dense_4/bias*
validate_shape(*
_output_shapes
:

main/dense_4/bias/Adam_1/readIdentitymain/dense_4/bias/Adam_1*
T0*$
_class
loc:@main/dense_4/bias*
_output_shapes
:
W
Adam/learning_rateConst*
valueB
 *Зб8*
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
 *wО?
Q
Adam/epsilonConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 

'Adam/update_main/dense/kernel/ApplyAdam	ApplyAdammain/dense/kernelmain/dense/kernel/Adammain/dense/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon;gradients/main/dense/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*$
_class
loc:@main/dense/kernel*
use_nesterov( *
_output_shapes
:	
ў
%Adam/update_main/dense/bias/ApplyAdam	ApplyAdammain/dense/biasmain/dense/bias/Adammain/dense/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon<gradients/main/dense/BiasAdd_grad/tuple/control_dependency_1*"
_class
loc:@main/dense/bias*
use_nesterov( *
_output_shapes	
:*
use_locking( *
T0

)Adam/update_main/dense_1/kernel/ApplyAdam	ApplyAdammain/dense_1/kernelmain/dense_1/kernel/Adammain/dense_1/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon=gradients/main/dense_1/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*&
_class
loc:@main/dense_1/kernel*
use_nesterov( *
_output_shapes
:	@

'Adam/update_main/dense_1/bias/ApplyAdam	ApplyAdammain/dense_1/biasmain/dense_1/bias/Adammain/dense_1/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon>gradients/main/dense_1/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*$
_class
loc:@main/dense_1/bias*
use_nesterov( *
_output_shapes	
:

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


'Adam/update_main/dense_2/bias/ApplyAdam	ApplyAdammain/dense_2/biasmain/dense_2/bias/Adammain/dense_2/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon>gradients/main/dense_2/BiasAdd_grad/tuple/control_dependency_1*$
_class
loc:@main/dense_2/bias*
use_nesterov( *
_output_shapes	
:*
use_locking( *
T0

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


'Adam/update_main/dense_3/bias/ApplyAdam	ApplyAdammain/dense_3/biasmain/dense_3/bias/Adammain/dense_3/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon>gradients/main/dense_3/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*$
_class
loc:@main/dense_3/bias*
use_nesterov( *
_output_shapes	
:

)Adam/update_main/dense_4/kernel/ApplyAdam	ApplyAdammain/dense_4/kernelmain/dense_4/kernel/Adammain/dense_4/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon=gradients/main/dense_4/MatMul_grad/tuple/control_dependency_1*
_output_shapes
:	*
use_locking( *
T0*&
_class
loc:@main/dense_4/kernel*
use_nesterov( 

'Adam/update_main/dense_4/bias/ApplyAdam	ApplyAdammain/dense_4/biasmain/dense_4/bias/Adammain/dense_4/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon>gradients/main/dense_4/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0*$
_class
loc:@main/dense_4/bias

Adam/mulMulbeta1_power/read
Adam/beta1&^Adam/update_main/dense/bias/ApplyAdam(^Adam/update_main/dense/kernel/ApplyAdam(^Adam/update_main/dense_1/bias/ApplyAdam*^Adam/update_main/dense_1/kernel/ApplyAdam(^Adam/update_main/dense_2/bias/ApplyAdam*^Adam/update_main/dense_2/kernel/ApplyAdam(^Adam/update_main/dense_3/bias/ApplyAdam*^Adam/update_main/dense_3/kernel/ApplyAdam(^Adam/update_main/dense_4/bias/ApplyAdam*^Adam/update_main/dense_4/kernel/ApplyAdam*"
_class
loc:@main/dense/bias*
_output_shapes
: *
T0

Adam/AssignAssignbeta1_powerAdam/mul*
validate_shape(*
_output_shapes
: *
use_locking( *
T0*"
_class
loc:@main/dense/bias


Adam/mul_1Mulbeta2_power/read
Adam/beta2&^Adam/update_main/dense/bias/ApplyAdam(^Adam/update_main/dense/kernel/ApplyAdam(^Adam/update_main/dense_1/bias/ApplyAdam*^Adam/update_main/dense_1/kernel/ApplyAdam(^Adam/update_main/dense_2/bias/ApplyAdam*^Adam/update_main/dense_2/kernel/ApplyAdam(^Adam/update_main/dense_3/bias/ApplyAdam*^Adam/update_main/dense_3/kernel/ApplyAdam(^Adam/update_main/dense_4/bias/ApplyAdam*^Adam/update_main/dense_4/kernel/ApplyAdam*
T0*"
_class
loc:@main/dense/bias*
_output_shapes
: 

Adam/Assign_1Assignbeta2_power
Adam/mul_1*
use_locking( *
T0*"
_class
loc:@main/dense/bias*
validate_shape(*
_output_shapes
: 
д
AdamNoOp^Adam/Assign^Adam/Assign_1&^Adam/update_main/dense/bias/ApplyAdam(^Adam/update_main/dense/kernel/ApplyAdam(^Adam/update_main/dense_1/bias/ApplyAdam*^Adam/update_main/dense_1/kernel/ApplyAdam(^Adam/update_main/dense_2/bias/ApplyAdam*^Adam/update_main/dense_2/kernel/ApplyAdam(^Adam/update_main/dense_3/bias/ApplyAdam*^Adam/update_main/dense_3/kernel/ApplyAdam(^Adam/update_main/dense_4/bias/ApplyAdam*^Adam/update_main/dense_4/kernel/ApplyAdam
И
AssignAssigntarget/dense/kernelmain/dense/kernel/read*&
_class
loc:@target/dense/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0
А
Assign_1Assigntarget/dense/biasmain/dense/bias/read*
T0*$
_class
loc:@target/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
Р
Assign_2Assigntarget/dense_1/kernelmain/dense_1/kernel/read*
_output_shapes
:	@*
use_locking(*
T0*(
_class
loc:@target/dense_1/kernel*
validate_shape(
Ж
Assign_3Assigntarget/dense_1/biasmain/dense_1/bias/read*&
_class
loc:@target/dense_1/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
С
Assign_4Assigntarget/dense_2/kernelmain/dense_2/kernel/read*
use_locking(*
T0*(
_class
loc:@target/dense_2/kernel*
validate_shape(* 
_output_shapes
:

Ж
Assign_5Assigntarget/dense_2/biasmain/dense_2/bias/read*
use_locking(*
T0*&
_class
loc:@target/dense_2/bias*
validate_shape(*
_output_shapes	
:
С
Assign_6Assigntarget/dense_3/kernelmain/dense_3/kernel/read*
use_locking(*
T0*(
_class
loc:@target/dense_3/kernel*
validate_shape(* 
_output_shapes
:

Ж
Assign_7Assigntarget/dense_3/biasmain/dense_3/bias/read*
use_locking(*
T0*&
_class
loc:@target/dense_3/bias*
validate_shape(*
_output_shapes	
:
Р
Assign_8Assigntarget/dense_4/kernelmain/dense_4/kernel/read*
use_locking(*
T0*(
_class
loc:@target/dense_4/kernel*
validate_shape(*
_output_shapes
:	
Е
Assign_9Assigntarget/dense_4/biasmain/dense_4/bias/read*
use_locking(*
T0*&
_class
loc:@target/dense_4/bias*
validate_shape(*
_output_shapes
:


initNoOp^beta1_power/Assign^beta2_power/Assign^main/dense/bias/Adam/Assign^main/dense/bias/Adam_1/Assign^main/dense/bias/Assign^main/dense/kernel/Adam/Assign ^main/dense/kernel/Adam_1/Assign^main/dense/kernel/Assign^main/dense_1/bias/Adam/Assign ^main/dense_1/bias/Adam_1/Assign^main/dense_1/bias/Assign ^main/dense_1/kernel/Adam/Assign"^main/dense_1/kernel/Adam_1/Assign^main/dense_1/kernel/Assign^main/dense_2/bias/Adam/Assign ^main/dense_2/bias/Adam_1/Assign^main/dense_2/bias/Assign ^main/dense_2/kernel/Adam/Assign"^main/dense_2/kernel/Adam_1/Assign^main/dense_2/kernel/Assign^main/dense_3/bias/Adam/Assign ^main/dense_3/bias/Adam_1/Assign^main/dense_3/bias/Assign ^main/dense_3/kernel/Adam/Assign"^main/dense_3/kernel/Adam_1/Assign^main/dense_3/kernel/Assign^main/dense_4/bias/Adam/Assign ^main/dense_4/bias/Adam_1/Assign^main/dense_4/bias/Assign ^main/dense_4/kernel/Adam/Assign"^main/dense_4/kernel/Adam_1/Assign^main/dense_4/kernel/Assign^target/dense/bias/Assign^target/dense/kernel/Assign^target/dense_1/bias/Assign^target/dense_1/kernel/Assign^target/dense_2/bias/Assign^target/dense_2/kernel/Assign^target/dense_3/bias/Assign^target/dense_3/kernel/Assign^target/dense_4/bias/Assign^target/dense_4/kernel/Assign
R
Placeholder_4Placeholder*
dtype0*
_output_shapes
:*
shape:
R
reward/tagsConst*
_output_shapes
: *
valueB Breward*
dtype0
T
rewardScalarSummaryreward/tagsPlaceholder_4*
T0*
_output_shapes
: 
K
Merge/MergeSummaryMergeSummaryreward*
_output_shapes
: *
N"oВЏх     ьW	`ькфжAJиЁ
 я
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
2	
ю
	ApplyAdam
var"T	
m"T	
v"T
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
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

2	

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

2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	
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
2	
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

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
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring *	1.9.0-rc12v1.9.0-rc0-35-g17d6639b55С
n
PlaceholderPlaceholder*
dtype0*'
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
p
Placeholder_1Placeholder*
dtype0*'
_output_shapes
:џџџџџџџџџ *
shape:џџџџџџџџџ 
p
Placeholder_2Placeholder*
shape:џџџџџџџџџ *
dtype0*'
_output_shapes
:џџџџџџџџџ 
p
Placeholder_3Placeholder*
dtype0*'
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
d
main/Tile/multiplesConst*
dtype0*
_output_shapes
:*
valueB"       
x
	main/TileTilePlaceholdermain/Tile/multiples*

Tmultiples0*
T0*(
_output_shapes
:џџџџџџџџџ
c
main/Reshape/shapeConst*
valueB"џџџџ   *
dtype0*
_output_shapes
:
v
main/ReshapeReshape	main/Tilemain/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Љ
2main/dense/kernel/Initializer/random_uniform/shapeConst*$
_class
loc:@main/dense/kernel*
valueB"      *
dtype0*
_output_shapes
:

0main/dense/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *$
_class
loc:@main/dense/kernel*
valueB
 *JQZО*
dtype0

0main/dense/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *$
_class
loc:@main/dense/kernel*
valueB
 *JQZ>
ѕ
:main/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform2main/dense/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	*

seed *
T0*$
_class
loc:@main/dense/kernel*
seed2 
т
0main/dense/kernel/Initializer/random_uniform/subSub0main/dense/kernel/Initializer/random_uniform/max0main/dense/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*$
_class
loc:@main/dense/kernel
ѕ
0main/dense/kernel/Initializer/random_uniform/mulMul:main/dense/kernel/Initializer/random_uniform/RandomUniform0main/dense/kernel/Initializer/random_uniform/sub*
_output_shapes
:	*
T0*$
_class
loc:@main/dense/kernel
ч
,main/dense/kernel/Initializer/random_uniformAdd0main/dense/kernel/Initializer/random_uniform/mul0main/dense/kernel/Initializer/random_uniform/min*
_output_shapes
:	*
T0*$
_class
loc:@main/dense/kernel
­
main/dense/kernel
VariableV2*
shape:	*
dtype0*
_output_shapes
:	*
shared_name *$
_class
loc:@main/dense/kernel*
	container 
м
main/dense/kernel/AssignAssignmain/dense/kernel,main/dense/kernel/Initializer/random_uniform*
use_locking(*
T0*$
_class
loc:@main/dense/kernel*
validate_shape(*
_output_shapes
:	

main/dense/kernel/readIdentitymain/dense/kernel*
T0*$
_class
loc:@main/dense/kernel*
_output_shapes
:	

!main/dense/bias/Initializer/zerosConst*"
_class
loc:@main/dense/bias*
valueB*    *
dtype0*
_output_shapes	
:
Ё
main/dense/bias
VariableV2*
shared_name *"
_class
loc:@main/dense/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
Ч
main/dense/bias/AssignAssignmain/dense/bias!main/dense/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*"
_class
loc:@main/dense/bias
{
main/dense/bias/readIdentitymain/dense/bias*
T0*"
_class
loc:@main/dense/bias*
_output_shapes	
:

main/dense/MatMulMatMulmain/Reshapemain/dense/kernel/read*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0

main/dense/BiasAddBiasAddmain/dense/MatMulmain/dense/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:џџџџџџџџџ
^
main/dense/SeluSelumain/dense/BiasAdd*(
_output_shapes
:џџџџџџџџџ*
T0
e
main/Reshape_1/shapeConst*
valueB"џџџџ   *
dtype0*
_output_shapes
:
~
main/Reshape_1ReshapePlaceholder_1main/Reshape_1/shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
п

main/ConstConst*
dtype0*
_output_shapes

:@*
valueB@"    лI@лЩ@фЫAлIAбS{AфЫAпэЏAлЩAж1тAбSћAц:
BфЫBт\#Bпэ/Bн~<BлIBи UBж1bBдТnBбS{BgђBц:BeBфЫBcBт\ЃB`ЅЉBпэЏB^6ЖBн~МB\ЧТBлЩBYXЯBи еBWщлBж1тBUzшBдТюBRѕBбSћB(Ю CgђCЇCц:
C&_CeCЅЇCфЫC#№CcCЂ8 Cт\#C!&C`Ѕ)C Щ,Cпэ/C3C^66CZ9Cн~<CЃ?C\ЧBCыEC

main/MatMulMatMulmain/Reshape_1
main/Const*'
_output_shapes
:џџџџџџџџџ@*
transpose_a( *
transpose_b( *
T0
N
main/CosCosmain/MatMul*
T0*'
_output_shapes
:џџџџџџџџџ@
­
4main/dense_1/kernel/Initializer/random_uniform/shapeConst*&
_class
loc:@main/dense_1/kernel*
valueB"@      *
dtype0*
_output_shapes
:

2main/dense_1/kernel/Initializer/random_uniform/minConst*&
_class
loc:@main/dense_1/kernel*
valueB
 *ѓ5О*
dtype0*
_output_shapes
: 

2main/dense_1/kernel/Initializer/random_uniform/maxConst*&
_class
loc:@main/dense_1/kernel*
valueB
 *ѓ5>*
dtype0*
_output_shapes
: 
ћ
<main/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform4main/dense_1/kernel/Initializer/random_uniform/shape*
seed2 *
dtype0*
_output_shapes
:	@*

seed *
T0*&
_class
loc:@main/dense_1/kernel
ъ
2main/dense_1/kernel/Initializer/random_uniform/subSub2main/dense_1/kernel/Initializer/random_uniform/max2main/dense_1/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*&
_class
loc:@main/dense_1/kernel
§
2main/dense_1/kernel/Initializer/random_uniform/mulMul<main/dense_1/kernel/Initializer/random_uniform/RandomUniform2main/dense_1/kernel/Initializer/random_uniform/sub*
T0*&
_class
loc:@main/dense_1/kernel*
_output_shapes
:	@
я
.main/dense_1/kernel/Initializer/random_uniformAdd2main/dense_1/kernel/Initializer/random_uniform/mul2main/dense_1/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@main/dense_1/kernel*
_output_shapes
:	@
Б
main/dense_1/kernel
VariableV2*
dtype0*
_output_shapes
:	@*
shared_name *&
_class
loc:@main/dense_1/kernel*
	container *
shape:	@
ф
main/dense_1/kernel/AssignAssignmain/dense_1/kernel.main/dense_1/kernel/Initializer/random_uniform*
T0*&
_class
loc:@main/dense_1/kernel*
validate_shape(*
_output_shapes
:	@*
use_locking(

main/dense_1/kernel/readIdentitymain/dense_1/kernel*
_output_shapes
:	@*
T0*&
_class
loc:@main/dense_1/kernel

#main/dense_1/bias/Initializer/zerosConst*$
_class
loc:@main/dense_1/bias*
valueB*    *
dtype0*
_output_shapes	
:
Ѕ
main/dense_1/bias
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *$
_class
loc:@main/dense_1/bias*
	container *
shape:
Я
main/dense_1/bias/AssignAssignmain/dense_1/bias#main/dense_1/bias/Initializer/zeros*$
_class
loc:@main/dense_1/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0

main/dense_1/bias/readIdentitymain/dense_1/bias*
T0*$
_class
loc:@main/dense_1/bias*
_output_shapes	
:

main/dense_1/MatMulMatMulmain/Cosmain/dense_1/kernel/read*
transpose_b( *
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( 

main/dense_1/BiasAddBiasAddmain/dense_1/MatMulmain/dense_1/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:џџџџџџџџџ
b
main/dense_1/ReluRelumain/dense_1/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
f
main/MulMulmain/dense/Selumain/dense_1/Relu*
T0*(
_output_shapes
:џџџџџџџџџ
­
4main/dense_2/kernel/Initializer/random_uniform/shapeConst*&
_class
loc:@main/dense_2/kernel*
valueB"      *
dtype0*
_output_shapes
:

2main/dense_2/kernel/Initializer/random_uniform/minConst*&
_class
loc:@main/dense_2/kernel*
valueB
 *јKЦН*
dtype0*
_output_shapes
: 

2main/dense_2/kernel/Initializer/random_uniform/maxConst*&
_class
loc:@main/dense_2/kernel*
valueB
 *јKЦ=*
dtype0*
_output_shapes
: 
ќ
<main/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform4main/dense_2/kernel/Initializer/random_uniform/shape*
T0*&
_class
loc:@main/dense_2/kernel*
seed2 *
dtype0* 
_output_shapes
:
*

seed 
ъ
2main/dense_2/kernel/Initializer/random_uniform/subSub2main/dense_2/kernel/Initializer/random_uniform/max2main/dense_2/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@main/dense_2/kernel*
_output_shapes
: 
ў
2main/dense_2/kernel/Initializer/random_uniform/mulMul<main/dense_2/kernel/Initializer/random_uniform/RandomUniform2main/dense_2/kernel/Initializer/random_uniform/sub*
T0*&
_class
loc:@main/dense_2/kernel* 
_output_shapes
:

№
.main/dense_2/kernel/Initializer/random_uniformAdd2main/dense_2/kernel/Initializer/random_uniform/mul2main/dense_2/kernel/Initializer/random_uniform/min*&
_class
loc:@main/dense_2/kernel* 
_output_shapes
:
*
T0
Г
main/dense_2/kernel
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *&
_class
loc:@main/dense_2/kernel*
	container *
shape:

х
main/dense_2/kernel/AssignAssignmain/dense_2/kernel.main/dense_2/kernel/Initializer/random_uniform*
T0*&
_class
loc:@main/dense_2/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(

main/dense_2/kernel/readIdentitymain/dense_2/kernel*
T0*&
_class
loc:@main/dense_2/kernel* 
_output_shapes
:


#main/dense_2/bias/Initializer/zerosConst*$
_class
loc:@main/dense_2/bias*
valueB*    *
dtype0*
_output_shapes	
:
Ѕ
main/dense_2/bias
VariableV2*
shared_name *$
_class
loc:@main/dense_2/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
Я
main/dense_2/bias/AssignAssignmain/dense_2/bias#main/dense_2/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*$
_class
loc:@main/dense_2/bias

main/dense_2/bias/readIdentitymain/dense_2/bias*
T0*$
_class
loc:@main/dense_2/bias*
_output_shapes	
:

main/dense_2/MatMulMatMulmain/Mulmain/dense_2/kernel/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 

main/dense_2/BiasAddBiasAddmain/dense_2/MatMulmain/dense_2/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:џџџџџџџџџ
b
main/dense_2/ReluRelumain/dense_2/BiasAdd*(
_output_shapes
:џџџџџџџџџ*
T0
­
4main/dense_3/kernel/Initializer/random_uniform/shapeConst*&
_class
loc:@main/dense_3/kernel*
valueB"      *
dtype0*
_output_shapes
:

2main/dense_3/kernel/Initializer/random_uniform/minConst*&
_class
loc:@main/dense_3/kernel*
valueB
 *јKЦН*
dtype0*
_output_shapes
: 

2main/dense_3/kernel/Initializer/random_uniform/maxConst*&
_class
loc:@main/dense_3/kernel*
valueB
 *јKЦ=*
dtype0*
_output_shapes
: 
ќ
<main/dense_3/kernel/Initializer/random_uniform/RandomUniformRandomUniform4main/dense_3/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
*

seed *
T0*&
_class
loc:@main/dense_3/kernel*
seed2 
ъ
2main/dense_3/kernel/Initializer/random_uniform/subSub2main/dense_3/kernel/Initializer/random_uniform/max2main/dense_3/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@main/dense_3/kernel*
_output_shapes
: 
ў
2main/dense_3/kernel/Initializer/random_uniform/mulMul<main/dense_3/kernel/Initializer/random_uniform/RandomUniform2main/dense_3/kernel/Initializer/random_uniform/sub*
T0*&
_class
loc:@main/dense_3/kernel* 
_output_shapes
:

№
.main/dense_3/kernel/Initializer/random_uniformAdd2main/dense_3/kernel/Initializer/random_uniform/mul2main/dense_3/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@main/dense_3/kernel* 
_output_shapes
:

Г
main/dense_3/kernel
VariableV2*
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *&
_class
loc:@main/dense_3/kernel*
	container 
х
main/dense_3/kernel/AssignAssignmain/dense_3/kernel.main/dense_3/kernel/Initializer/random_uniform*
use_locking(*
T0*&
_class
loc:@main/dense_3/kernel*
validate_shape(* 
_output_shapes
:


main/dense_3/kernel/readIdentitymain/dense_3/kernel* 
_output_shapes
:
*
T0*&
_class
loc:@main/dense_3/kernel

#main/dense_3/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:*$
_class
loc:@main/dense_3/bias*
valueB*    
Ѕ
main/dense_3/bias
VariableV2*
shared_name *$
_class
loc:@main/dense_3/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
Я
main/dense_3/bias/AssignAssignmain/dense_3/bias#main/dense_3/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*$
_class
loc:@main/dense_3/bias

main/dense_3/bias/readIdentitymain/dense_3/bias*
T0*$
_class
loc:@main/dense_3/bias*
_output_shapes	
:
Ѓ
main/dense_3/MatMulMatMulmain/dense_2/Relumain/dense_3/kernel/read*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0

main/dense_3/BiasAddBiasAddmain/dense_3/MatMulmain/dense_3/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:џџџџџџџџџ
b
main/dense_3/ReluRelumain/dense_3/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
­
4main/dense_4/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*&
_class
loc:@main/dense_4/kernel*
valueB"      

2main/dense_4/kernel/Initializer/random_uniform/minConst*&
_class
loc:@main/dense_4/kernel*
valueB
 *§[О*
dtype0*
_output_shapes
: 

2main/dense_4/kernel/Initializer/random_uniform/maxConst*&
_class
loc:@main/dense_4/kernel*
valueB
 *§[>*
dtype0*
_output_shapes
: 
ћ
<main/dense_4/kernel/Initializer/random_uniform/RandomUniformRandomUniform4main/dense_4/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	*

seed *
T0*&
_class
loc:@main/dense_4/kernel*
seed2 
ъ
2main/dense_4/kernel/Initializer/random_uniform/subSub2main/dense_4/kernel/Initializer/random_uniform/max2main/dense_4/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@main/dense_4/kernel*
_output_shapes
: 
§
2main/dense_4/kernel/Initializer/random_uniform/mulMul<main/dense_4/kernel/Initializer/random_uniform/RandomUniform2main/dense_4/kernel/Initializer/random_uniform/sub*
T0*&
_class
loc:@main/dense_4/kernel*
_output_shapes
:	
я
.main/dense_4/kernel/Initializer/random_uniformAdd2main/dense_4/kernel/Initializer/random_uniform/mul2main/dense_4/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@main/dense_4/kernel*
_output_shapes
:	
Б
main/dense_4/kernel
VariableV2*
dtype0*
_output_shapes
:	*
shared_name *&
_class
loc:@main/dense_4/kernel*
	container *
shape:	
ф
main/dense_4/kernel/AssignAssignmain/dense_4/kernel.main/dense_4/kernel/Initializer/random_uniform*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*&
_class
loc:@main/dense_4/kernel

main/dense_4/kernel/readIdentitymain/dense_4/kernel*
T0*&
_class
loc:@main/dense_4/kernel*
_output_shapes
:	

#main/dense_4/bias/Initializer/zerosConst*
_output_shapes
:*$
_class
loc:@main/dense_4/bias*
valueB*    *
dtype0
Ѓ
main/dense_4/bias
VariableV2*
dtype0*
_output_shapes
:*
shared_name *$
_class
loc:@main/dense_4/bias*
	container *
shape:
Ю
main/dense_4/bias/AssignAssignmain/dense_4/bias#main/dense_4/bias/Initializer/zeros*$
_class
loc:@main/dense_4/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0

main/dense_4/bias/readIdentitymain/dense_4/bias*
_output_shapes
:*
T0*$
_class
loc:@main/dense_4/bias
Ђ
main/dense_4/MatMulMatMulmain/dense_3/Relumain/dense_4/kernel/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 

main/dense_4/BiasAddBiasAddmain/dense_4/MatMulmain/dense_4/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ
N
main/Const_1Const*
_output_shapes
: *
value	B : *
dtype0
V
main/split/split_dimConst*
value	B : *
dtype0*
_output_shapes
: 
в

main/splitSplitmain/split/split_dimmain/dense_4/BiasAdd*
T0*і
_output_shapesу
р:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split 
З
main/transpose/xPack
main/splitmain/split:1main/split:2main/split:3main/split:4main/split:5main/split:6main/split:7main/split:8main/split:9main/split:10main/split:11main/split:12main/split:13main/split:14main/split:15main/split:16main/split:17main/split:18main/split:19main/split:20main/split:21main/split:22main/split:23main/split:24main/split:25main/split:26main/split:27main/split:28main/split:29main/split:30main/split:31*+
_output_shapes
: џџџџџџџџџ*
T0*

axis *
N 
h
main/transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:

main/transpose	Transposemain/transpose/xmain/transpose/perm*+
_output_shapes
: џџџџџџџџџ*
Tperm0*
T0
f
target/Tile/multiplesConst*
valueB"       *
dtype0*
_output_shapes
:
|
target/TileTilePlaceholdertarget/Tile/multiples*(
_output_shapes
:џџџџџџџџџ*

Tmultiples0*
T0
e
target/Reshape/shapeConst*
_output_shapes
:*
valueB"џџџџ   *
dtype0
|
target/ReshapeReshapetarget/Tiletarget/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
­
4target/dense/kernel/Initializer/random_uniform/shapeConst*&
_class
loc:@target/dense/kernel*
valueB"      *
dtype0*
_output_shapes
:

2target/dense/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *&
_class
loc:@target/dense/kernel*
valueB
 *JQZО

2target/dense/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *&
_class
loc:@target/dense/kernel*
valueB
 *JQZ>*
dtype0
ћ
<target/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform4target/dense/kernel/Initializer/random_uniform/shape*
T0*&
_class
loc:@target/dense/kernel*
seed2 *
dtype0*
_output_shapes
:	*

seed 
ъ
2target/dense/kernel/Initializer/random_uniform/subSub2target/dense/kernel/Initializer/random_uniform/max2target/dense/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@target/dense/kernel*
_output_shapes
: 
§
2target/dense/kernel/Initializer/random_uniform/mulMul<target/dense/kernel/Initializer/random_uniform/RandomUniform2target/dense/kernel/Initializer/random_uniform/sub*
_output_shapes
:	*
T0*&
_class
loc:@target/dense/kernel
я
.target/dense/kernel/Initializer/random_uniformAdd2target/dense/kernel/Initializer/random_uniform/mul2target/dense/kernel/Initializer/random_uniform/min*&
_class
loc:@target/dense/kernel*
_output_shapes
:	*
T0
Б
target/dense/kernel
VariableV2*
dtype0*
_output_shapes
:	*
shared_name *&
_class
loc:@target/dense/kernel*
	container *
shape:	
ф
target/dense/kernel/AssignAssigntarget/dense/kernel.target/dense/kernel/Initializer/random_uniform*
use_locking(*
T0*&
_class
loc:@target/dense/kernel*
validate_shape(*
_output_shapes
:	

target/dense/kernel/readIdentitytarget/dense/kernel*
T0*&
_class
loc:@target/dense/kernel*
_output_shapes
:	

#target/dense/bias/Initializer/zerosConst*$
_class
loc:@target/dense/bias*
valueB*    *
dtype0*
_output_shapes	
:
Ѕ
target/dense/bias
VariableV2*
shared_name *$
_class
loc:@target/dense/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
Я
target/dense/bias/AssignAssigntarget/dense/bias#target/dense/bias/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*$
_class
loc:@target/dense/bias*
validate_shape(

target/dense/bias/readIdentitytarget/dense/bias*
T0*$
_class
loc:@target/dense/bias*
_output_shapes	
:
 
target/dense/MatMulMatMultarget/Reshapetarget/dense/kernel/read*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0

target/dense/BiasAddBiasAddtarget/dense/MatMultarget/dense/bias/read*
data_formatNHWC*(
_output_shapes
:џџџџџџџџџ*
T0
b
target/dense/SeluSelutarget/dense/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
g
target/Reshape_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"џџџџ   

target/Reshape_1ReshapePlaceholder_1target/Reshape_1/shape*
Tshape0*'
_output_shapes
:џџџџџџџџџ*
T0
с
target/ConstConst*
_output_shapes

:@*
valueB@"    лI@лЩ@фЫAлIAбS{AфЫAпэЏAлЩAж1тAбSћAц:
BфЫBт\#Bпэ/Bн~<BлIBи UBж1bBдТnBбS{BgђBц:BeBфЫBcBт\ЃB`ЅЉBпэЏB^6ЖBн~МB\ЧТBлЩBYXЯBи еBWщлBж1тBUzшBдТюBRѕBбSћB(Ю CgђCЇCц:
C&_CeCЅЇCфЫC#№CcCЂ8 Cт\#C!&C`Ѕ)C Щ,Cпэ/C3C^66CZ9Cн~<CЃ?C\ЧBCыEC*
dtype0

target/MatMulMatMultarget/Reshape_1target/Const*
T0*'
_output_shapes
:џџџџџџџџџ@*
transpose_a( *
transpose_b( 
R

target/CosCostarget/MatMul*'
_output_shapes
:џџџџџџџџџ@*
T0
Б
6target/dense_1/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*(
_class
loc:@target/dense_1/kernel*
valueB"@      
Ѓ
4target/dense_1/kernel/Initializer/random_uniform/minConst*(
_class
loc:@target/dense_1/kernel*
valueB
 *ѓ5О*
dtype0*
_output_shapes
: 
Ѓ
4target/dense_1/kernel/Initializer/random_uniform/maxConst*(
_class
loc:@target/dense_1/kernel*
valueB
 *ѓ5>*
dtype0*
_output_shapes
: 

>target/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform6target/dense_1/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	@*

seed *
T0*(
_class
loc:@target/dense_1/kernel*
seed2 
ђ
4target/dense_1/kernel/Initializer/random_uniform/subSub4target/dense_1/kernel/Initializer/random_uniform/max4target/dense_1/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@target/dense_1/kernel*
_output_shapes
: 

4target/dense_1/kernel/Initializer/random_uniform/mulMul>target/dense_1/kernel/Initializer/random_uniform/RandomUniform4target/dense_1/kernel/Initializer/random_uniform/sub*
_output_shapes
:	@*
T0*(
_class
loc:@target/dense_1/kernel
ї
0target/dense_1/kernel/Initializer/random_uniformAdd4target/dense_1/kernel/Initializer/random_uniform/mul4target/dense_1/kernel/Initializer/random_uniform/min*
_output_shapes
:	@*
T0*(
_class
loc:@target/dense_1/kernel
Е
target/dense_1/kernel
VariableV2*
dtype0*
_output_shapes
:	@*
shared_name *(
_class
loc:@target/dense_1/kernel*
	container *
shape:	@
ь
target/dense_1/kernel/AssignAssigntarget/dense_1/kernel0target/dense_1/kernel/Initializer/random_uniform*
_output_shapes
:	@*
use_locking(*
T0*(
_class
loc:@target/dense_1/kernel*
validate_shape(

target/dense_1/kernel/readIdentitytarget/dense_1/kernel*
T0*(
_class
loc:@target/dense_1/kernel*
_output_shapes
:	@

%target/dense_1/bias/Initializer/zerosConst*&
_class
loc:@target/dense_1/bias*
valueB*    *
dtype0*
_output_shapes	
:
Љ
target/dense_1/bias
VariableV2*
shared_name *&
_class
loc:@target/dense_1/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
з
target/dense_1/bias/AssignAssigntarget/dense_1/bias%target/dense_1/bias/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*&
_class
loc:@target/dense_1/bias*
validate_shape(

target/dense_1/bias/readIdentitytarget/dense_1/bias*
T0*&
_class
loc:@target/dense_1/bias*
_output_shapes	
:
 
target/dense_1/MatMulMatMul
target/Costarget/dense_1/kernel/read*
transpose_b( *
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( 

target/dense_1/BiasAddBiasAddtarget/dense_1/MatMultarget/dense_1/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:џџџџџџџџџ
f
target/dense_1/ReluRelutarget/dense_1/BiasAdd*(
_output_shapes
:џџџџџџџџџ*
T0
l

target/MulMultarget/dense/Selutarget/dense_1/Relu*(
_output_shapes
:џџџџџџџџџ*
T0
Б
6target/dense_2/kernel/Initializer/random_uniform/shapeConst*(
_class
loc:@target/dense_2/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ѓ
4target/dense_2/kernel/Initializer/random_uniform/minConst*(
_class
loc:@target/dense_2/kernel*
valueB
 *јKЦН*
dtype0*
_output_shapes
: 
Ѓ
4target/dense_2/kernel/Initializer/random_uniform/maxConst*(
_class
loc:@target/dense_2/kernel*
valueB
 *јKЦ=*
dtype0*
_output_shapes
: 

>target/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform6target/dense_2/kernel/Initializer/random_uniform/shape*
seed2 *
dtype0* 
_output_shapes
:
*

seed *
T0*(
_class
loc:@target/dense_2/kernel
ђ
4target/dense_2/kernel/Initializer/random_uniform/subSub4target/dense_2/kernel/Initializer/random_uniform/max4target/dense_2/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@target/dense_2/kernel*
_output_shapes
: 

4target/dense_2/kernel/Initializer/random_uniform/mulMul>target/dense_2/kernel/Initializer/random_uniform/RandomUniform4target/dense_2/kernel/Initializer/random_uniform/sub*
T0*(
_class
loc:@target/dense_2/kernel* 
_output_shapes
:

ј
0target/dense_2/kernel/Initializer/random_uniformAdd4target/dense_2/kernel/Initializer/random_uniform/mul4target/dense_2/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@target/dense_2/kernel* 
_output_shapes
:

З
target/dense_2/kernel
VariableV2*
shared_name *(
_class
loc:@target/dense_2/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:

э
target/dense_2/kernel/AssignAssigntarget/dense_2/kernel0target/dense_2/kernel/Initializer/random_uniform* 
_output_shapes
:
*
use_locking(*
T0*(
_class
loc:@target/dense_2/kernel*
validate_shape(

target/dense_2/kernel/readIdentitytarget/dense_2/kernel*
T0*(
_class
loc:@target/dense_2/kernel* 
_output_shapes
:


%target/dense_2/bias/Initializer/zerosConst*&
_class
loc:@target/dense_2/bias*
valueB*    *
dtype0*
_output_shapes	
:
Љ
target/dense_2/bias
VariableV2*
_output_shapes	
:*
shared_name *&
_class
loc:@target/dense_2/bias*
	container *
shape:*
dtype0
з
target/dense_2/bias/AssignAssigntarget/dense_2/bias%target/dense_2/bias/Initializer/zeros*
use_locking(*
T0*&
_class
loc:@target/dense_2/bias*
validate_shape(*
_output_shapes	
:

target/dense_2/bias/readIdentitytarget/dense_2/bias*
T0*&
_class
loc:@target/dense_2/bias*
_output_shapes	
:
 
target/dense_2/MatMulMatMul
target/Multarget/dense_2/kernel/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 

target/dense_2/BiasAddBiasAddtarget/dense_2/MatMultarget/dense_2/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:џџџџџџџџџ
f
target/dense_2/ReluRelutarget/dense_2/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
Б
6target/dense_3/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*(
_class
loc:@target/dense_3/kernel*
valueB"      *
dtype0
Ѓ
4target/dense_3/kernel/Initializer/random_uniform/minConst*(
_class
loc:@target/dense_3/kernel*
valueB
 *јKЦН*
dtype0*
_output_shapes
: 
Ѓ
4target/dense_3/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *(
_class
loc:@target/dense_3/kernel*
valueB
 *јKЦ=

>target/dense_3/kernel/Initializer/random_uniform/RandomUniformRandomUniform6target/dense_3/kernel/Initializer/random_uniform/shape*(
_class
loc:@target/dense_3/kernel*
seed2 *
dtype0* 
_output_shapes
:
*

seed *
T0
ђ
4target/dense_3/kernel/Initializer/random_uniform/subSub4target/dense_3/kernel/Initializer/random_uniform/max4target/dense_3/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@target/dense_3/kernel*
_output_shapes
: 

4target/dense_3/kernel/Initializer/random_uniform/mulMul>target/dense_3/kernel/Initializer/random_uniform/RandomUniform4target/dense_3/kernel/Initializer/random_uniform/sub*
T0*(
_class
loc:@target/dense_3/kernel* 
_output_shapes
:

ј
0target/dense_3/kernel/Initializer/random_uniformAdd4target/dense_3/kernel/Initializer/random_uniform/mul4target/dense_3/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@target/dense_3/kernel* 
_output_shapes
:

З
target/dense_3/kernel
VariableV2*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *(
_class
loc:@target/dense_3/kernel
э
target/dense_3/kernel/AssignAssigntarget/dense_3/kernel0target/dense_3/kernel/Initializer/random_uniform*(
_class
loc:@target/dense_3/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0

target/dense_3/kernel/readIdentitytarget/dense_3/kernel*
T0*(
_class
loc:@target/dense_3/kernel* 
_output_shapes
:


%target/dense_3/bias/Initializer/zerosConst*&
_class
loc:@target/dense_3/bias*
valueB*    *
dtype0*
_output_shapes	
:
Љ
target/dense_3/bias
VariableV2*
shared_name *&
_class
loc:@target/dense_3/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
з
target/dense_3/bias/AssignAssigntarget/dense_3/bias%target/dense_3/bias/Initializer/zeros*
use_locking(*
T0*&
_class
loc:@target/dense_3/bias*
validate_shape(*
_output_shapes	
:

target/dense_3/bias/readIdentitytarget/dense_3/bias*
_output_shapes	
:*
T0*&
_class
loc:@target/dense_3/bias
Љ
target/dense_3/MatMulMatMultarget/dense_2/Relutarget/dense_3/kernel/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 

target/dense_3/BiasAddBiasAddtarget/dense_3/MatMultarget/dense_3/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:џџџџџџџџџ
f
target/dense_3/ReluRelutarget/dense_3/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
Б
6target/dense_4/kernel/Initializer/random_uniform/shapeConst*(
_class
loc:@target/dense_4/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ѓ
4target/dense_4/kernel/Initializer/random_uniform/minConst*(
_class
loc:@target/dense_4/kernel*
valueB
 *§[О*
dtype0*
_output_shapes
: 
Ѓ
4target/dense_4/kernel/Initializer/random_uniform/maxConst*(
_class
loc:@target/dense_4/kernel*
valueB
 *§[>*
dtype0*
_output_shapes
: 

>target/dense_4/kernel/Initializer/random_uniform/RandomUniformRandomUniform6target/dense_4/kernel/Initializer/random_uniform/shape*
T0*(
_class
loc:@target/dense_4/kernel*
seed2 *
dtype0*
_output_shapes
:	*

seed 
ђ
4target/dense_4/kernel/Initializer/random_uniform/subSub4target/dense_4/kernel/Initializer/random_uniform/max4target/dense_4/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@target/dense_4/kernel*
_output_shapes
: 

4target/dense_4/kernel/Initializer/random_uniform/mulMul>target/dense_4/kernel/Initializer/random_uniform/RandomUniform4target/dense_4/kernel/Initializer/random_uniform/sub*
T0*(
_class
loc:@target/dense_4/kernel*
_output_shapes
:	
ї
0target/dense_4/kernel/Initializer/random_uniformAdd4target/dense_4/kernel/Initializer/random_uniform/mul4target/dense_4/kernel/Initializer/random_uniform/min*(
_class
loc:@target/dense_4/kernel*
_output_shapes
:	*
T0
Е
target/dense_4/kernel
VariableV2*(
_class
loc:@target/dense_4/kernel*
	container *
shape:	*
dtype0*
_output_shapes
:	*
shared_name 
ь
target/dense_4/kernel/AssignAssigntarget/dense_4/kernel0target/dense_4/kernel/Initializer/random_uniform*
use_locking(*
T0*(
_class
loc:@target/dense_4/kernel*
validate_shape(*
_output_shapes
:	

target/dense_4/kernel/readIdentitytarget/dense_4/kernel*
T0*(
_class
loc:@target/dense_4/kernel*
_output_shapes
:	

%target/dense_4/bias/Initializer/zerosConst*&
_class
loc:@target/dense_4/bias*
valueB*    *
dtype0*
_output_shapes
:
Ї
target/dense_4/bias
VariableV2*
_output_shapes
:*
shared_name *&
_class
loc:@target/dense_4/bias*
	container *
shape:*
dtype0
ж
target/dense_4/bias/AssignAssigntarget/dense_4/bias%target/dense_4/bias/Initializer/zeros*
use_locking(*
T0*&
_class
loc:@target/dense_4/bias*
validate_shape(*
_output_shapes
:

target/dense_4/bias/readIdentitytarget/dense_4/bias*
T0*&
_class
loc:@target/dense_4/bias*
_output_shapes
:
Ј
target/dense_4/MatMulMatMultarget/dense_3/Relutarget/dense_4/kernel/read*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0

target/dense_4/BiasAddBiasAddtarget/dense_4/MatMultarget/dense_4/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ
P
target/Const_1Const*
_output_shapes
: *
value	B : *
dtype0
X
target/split/split_dimConst*
value	B : *
dtype0*
_output_shapes
: 
и
target/splitSplittarget/split/split_dimtarget/dense_4/BiasAdd*і
_output_shapesу
р:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split *
T0
љ
target/transpose/xPacktarget/splittarget/split:1target/split:2target/split:3target/split:4target/split:5target/split:6target/split:7target/split:8target/split:9target/split:10target/split:11target/split:12target/split:13target/split:14target/split:15target/split:16target/split:17target/split:18target/split:19target/split:20target/split:21target/split:22target/split:23target/split:24target/split:25target/split:26target/split:27target/split:28target/split:29target/split:30target/split:31*
T0*

axis *
N *+
_output_shapes
: џџџџџџџџџ
j
target/transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:

target/transpose	Transposetarget/transpose/xtarget/transpose/perm*
Tperm0*
T0*+
_output_shapes
: џџџџџџџџџ
Y
ExpandDims/dimConst*
_output_shapes
: *
valueB :
џџџџџџџџџ*
dtype0
y

ExpandDims
ExpandDimsPlaceholder_3ExpandDims/dim*
T0*+
_output_shapes
:џџџџџџџџџ*

Tdim0
\
mulMulmain/transpose
ExpandDims*+
_output_shapes
: џџџџџџџџџ*
T0
W
Sum/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
u
SumSummulSum/reduction_indices*'
_output_shapes
: џџџџџџџџџ*
	keep_dims( *

Tidx0*
T0
R
huber_loss/SubSubSumPlaceholder_2*
_output_shapes

:  *
T0
N
huber_loss/AbsAbshuber_loss/Sub*
_output_shapes

:  *
T0
Y
huber_loss/Minimum/yConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
l
huber_loss/MinimumMinimumhuber_loss/Abshuber_loss/Minimum/y*
T0*
_output_shapes

:  
d
huber_loss/Sub_1Subhuber_loss/Abshuber_loss/Minimum*
T0*
_output_shapes

:  
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

:  *
T0
b
huber_loss/Mul_1Mulhuber_loss/Consthuber_loss/Mul*
_output_shapes

:  *
T0
W
huber_loss/Mul_2/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
huber_loss/Mul_2Mulhuber_loss/Mul_2/xhuber_loss/Sub_1*
_output_shapes

:  *
T0
b
huber_loss/AddAddhuber_loss/Mul_1huber_loss/Mul_2*
_output_shapes

:  *
T0
l
'huber_loss/assert_broadcastable/weightsConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
p
-huber_loss/assert_broadcastable/weights/shapeConst*
dtype0*
_output_shapes
: *
valueB 
n
,huber_loss/assert_broadcastable/weights/rankConst*
value	B : *
dtype0*
_output_shapes
: 
}
,huber_loss/assert_broadcastable/values/shapeConst*
valueB"        *
dtype0*
_output_shapes
:
m
+huber_loss/assert_broadcastable/values/rankConst*
_output_shapes
: *
value	B :*
dtype0
C
;huber_loss/assert_broadcastable/static_scalar_check_successNoOp

huber_loss/ToFloat_3/xConst<^huber_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *  ?*
dtype0*
_output_shapes
: 
h
huber_loss/Mul_3Mulhuber_loss/Addhuber_loss/ToFloat_3/x*
T0*
_output_shapes

:  
J
sub/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
R
subSubsub/xPlaceholder_1*
T0*'
_output_shapes
:џџџџџџџџџ 
I
sub_1SubPlaceholder_2Sum*
T0*
_output_shapes

:  
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

:  
L
mul_1Mulsubhuber_loss/Mul_3*
_output_shapes

:  *
T0
V
mul_2MulPlaceholder_1huber_loss/Mul_3*
T0*
_output_shapes

:  
M
SelectSelectLessmul_1mul_2*
T0*
_output_shapes

:  
Y
Sum_1/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
o
Sum_1SumSelectSum_1/reduction_indices*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
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
 *  ?*
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

gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
c
gradients/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:

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

gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Const_1*
T0*
_output_shapes
: 
k
gradients/Sum_1_grad/ShapeConst*
valueB"        *
dtype0*
_output_shapes
:

gradients/Sum_1_grad/SizeConst*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
Ѓ
gradients/Sum_1_grad/addAddSum_1/reduction_indicesgradients/Sum_1_grad/Size*
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
_output_shapes
: 
Љ
gradients/Sum_1_grad/modFloorModgradients/Sum_1_grad/addgradients/Sum_1_grad/Size*
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
_output_shapes
: 

gradients/Sum_1_grad/Shape_1Const*
_output_shapes
: *-
_class#
!loc:@gradients/Sum_1_grad/Shape*
valueB *
dtype0

 gradients/Sum_1_grad/range/startConst*
dtype0*
_output_shapes
: *-
_class#
!loc:@gradients/Sum_1_grad/Shape*
value	B : 

 gradients/Sum_1_grad/range/deltaConst*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
й
gradients/Sum_1_grad/rangeRange gradients/Sum_1_grad/range/startgradients/Sum_1_grad/Size gradients/Sum_1_grad/range/delta*

Tidx0*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
_output_shapes
:

gradients/Sum_1_grad/Fill/valueConst*
_output_shapes
: *-
_class#
!loc:@gradients/Sum_1_grad/Shape*
value	B :*
dtype0
Т
gradients/Sum_1_grad/FillFillgradients/Sum_1_grad/Shape_1gradients/Sum_1_grad/Fill/value*
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape*

index_type0*
_output_shapes
: 

"gradients/Sum_1_grad/DynamicStitchDynamicStitchgradients/Sum_1_grad/rangegradients/Sum_1_grad/modgradients/Sum_1_grad/Shapegradients/Sum_1_grad/Fill*
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
N*#
_output_shapes
:џџџџџџџџџ

gradients/Sum_1_grad/Maximum/yConst*-
_class#
!loc:@gradients/Sum_1_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
Ш
gradients/Sum_1_grad/MaximumMaximum"gradients/Sum_1_grad/DynamicStitchgradients/Sum_1_grad/Maximum/y*#
_output_shapes
:џџџџџџџџџ*
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape
З
gradients/Sum_1_grad/floordivFloorDivgradients/Sum_1_grad/Shapegradients/Sum_1_grad/Maximum*
_output_shapes
:*
T0*-
_class#
!loc:@gradients/Sum_1_grad/Shape

gradients/Sum_1_grad/ReshapeReshapegradients/Mean_grad/truediv"gradients/Sum_1_grad/DynamicStitch*
_output_shapes
:*
T0*
Tshape0

gradients/Sum_1_grad/TileTilegradients/Sum_1_grad/Reshapegradients/Sum_1_grad/floordiv*
T0*
_output_shapes

:  *

Tmultiples0

0gradients/Select_grad/zeros_like/shape_as_tensorConst*
valueB"        *
dtype0*
_output_shapes
:
k
&gradients/Select_grad/zeros_like/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Н
 gradients/Select_grad/zeros_likeFill0gradients/Select_grad/zeros_like/shape_as_tensor&gradients/Select_grad/zeros_like/Const*
_output_shapes

:  *
T0*

index_type0

gradients/Select_grad/SelectSelectLessgradients/Sum_1_grad/Tile gradients/Select_grad/zeros_like*
T0*
_output_shapes

:  

gradients/Select_grad/Select_1SelectLess gradients/Select_grad/zeros_likegradients/Sum_1_grad/Tile*
T0*
_output_shapes

:  
n
&gradients/Select_grad/tuple/group_depsNoOp^gradients/Select_grad/Select^gradients/Select_grad/Select_1
л
.gradients/Select_grad/tuple/control_dependencyIdentitygradients/Select_grad/Select'^gradients/Select_grad/tuple/group_deps*
_output_shapes

:  *
T0*/
_class%
#!loc:@gradients/Select_grad/Select
с
0gradients/Select_grad/tuple/control_dependency_1Identitygradients/Select_grad/Select_1'^gradients/Select_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/Select_grad/Select_1*
_output_shapes

:  
]
gradients/mul_1_grad/ShapeShapesub*
T0*
out_type0*
_output_shapes
:
m
gradients/mul_1_grad/Shape_1Const*
valueB"        *
dtype0*
_output_shapes
:
К
*gradients/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_1_grad/Shapegradients/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

gradients/mul_1_grad/MulMul.gradients/Select_grad/tuple/control_dependencyhuber_loss/Mul_3*
_output_shapes

:  *
T0
Ѕ
gradients/mul_1_grad/SumSumgradients/mul_1_grad/Mul*gradients/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

gradients/mul_1_grad/ReshapeReshapegradients/mul_1_grad/Sumgradients/mul_1_grad/Shape*'
_output_shapes
:џџџџџџџџџ *
T0*
Tshape0

gradients/mul_1_grad/Mul_1Mulsub.gradients/Select_grad/tuple/control_dependency*
_output_shapes

:  *
T0
Ћ
gradients/mul_1_grad/Sum_1Sumgradients/mul_1_grad/Mul_1,gradients/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

gradients/mul_1_grad/Reshape_1Reshapegradients/mul_1_grad/Sum_1gradients/mul_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:  
m
%gradients/mul_1_grad/tuple/group_depsNoOp^gradients/mul_1_grad/Reshape^gradients/mul_1_grad/Reshape_1
т
-gradients/mul_1_grad/tuple/control_dependencyIdentitygradients/mul_1_grad/Reshape&^gradients/mul_1_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/mul_1_grad/Reshape*'
_output_shapes
:џџџџџџџџџ 
п
/gradients/mul_1_grad/tuple/control_dependency_1Identitygradients/mul_1_grad/Reshape_1&^gradients/mul_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/mul_1_grad/Reshape_1*
_output_shapes

:  
g
gradients/mul_2_grad/ShapeShapePlaceholder_1*
T0*
out_type0*
_output_shapes
:
m
gradients/mul_2_grad/Shape_1Const*
valueB"        *
dtype0*
_output_shapes
:
К
*gradients/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_2_grad/Shapegradients/mul_2_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0

gradients/mul_2_grad/MulMul0gradients/Select_grad/tuple/control_dependency_1huber_loss/Mul_3*
T0*
_output_shapes

:  
Ѕ
gradients/mul_2_grad/SumSumgradients/mul_2_grad/Mul*gradients/mul_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

gradients/mul_2_grad/ReshapeReshapegradients/mul_2_grad/Sumgradients/mul_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ 

gradients/mul_2_grad/Mul_1MulPlaceholder_10gradients/Select_grad/tuple/control_dependency_1*
T0*
_output_shapes

:  
Ћ
gradients/mul_2_grad/Sum_1Sumgradients/mul_2_grad/Mul_1,gradients/mul_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

gradients/mul_2_grad/Reshape_1Reshapegradients/mul_2_grad/Sum_1gradients/mul_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:  
m
%gradients/mul_2_grad/tuple/group_depsNoOp^gradients/mul_2_grad/Reshape^gradients/mul_2_grad/Reshape_1
т
-gradients/mul_2_grad/tuple/control_dependencyIdentitygradients/mul_2_grad/Reshape&^gradients/mul_2_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ *
T0*/
_class%
#!loc:@gradients/mul_2_grad/Reshape
п
/gradients/mul_2_grad/tuple/control_dependency_1Identitygradients/mul_2_grad/Reshape_1&^gradients/mul_2_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/mul_2_grad/Reshape_1*
_output_shapes

:  
н
gradients/AddNAddN/gradients/mul_1_grad/tuple/control_dependency_1/gradients/mul_2_grad/tuple/control_dependency_1*
T0*1
_class'
%#loc:@gradients/mul_1_grad/Reshape_1*
N*
_output_shapes

:  
v
%gradients/huber_loss/Mul_3_grad/ShapeConst*
valueB"        *
dtype0*
_output_shapes
:
j
'gradients/huber_loss/Mul_3_grad/Shape_1Const*
_output_shapes
: *
valueB *
dtype0
л
5gradients/huber_loss/Mul_3_grad/BroadcastGradientArgsBroadcastGradientArgs%gradients/huber_loss/Mul_3_grad/Shape'gradients/huber_loss/Mul_3_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
{
#gradients/huber_loss/Mul_3_grad/MulMulgradients/AddNhuber_loss/ToFloat_3/x*
T0*
_output_shapes

:  
Ц
#gradients/huber_loss/Mul_3_grad/SumSum#gradients/huber_loss/Mul_3_grad/Mul5gradients/huber_loss/Mul_3_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Е
'gradients/huber_loss/Mul_3_grad/ReshapeReshape#gradients/huber_loss/Mul_3_grad/Sum%gradients/huber_loss/Mul_3_grad/Shape*
Tshape0*
_output_shapes

:  *
T0
u
%gradients/huber_loss/Mul_3_grad/Mul_1Mulhuber_loss/Addgradients/AddN*
T0*
_output_shapes

:  
Ь
%gradients/huber_loss/Mul_3_grad/Sum_1Sum%gradients/huber_loss/Mul_3_grad/Mul_17gradients/huber_loss/Mul_3_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Г
)gradients/huber_loss/Mul_3_grad/Reshape_1Reshape%gradients/huber_loss/Mul_3_grad/Sum_1'gradients/huber_loss/Mul_3_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 

0gradients/huber_loss/Mul_3_grad/tuple/group_depsNoOp(^gradients/huber_loss/Mul_3_grad/Reshape*^gradients/huber_loss/Mul_3_grad/Reshape_1

8gradients/huber_loss/Mul_3_grad/tuple/control_dependencyIdentity'gradients/huber_loss/Mul_3_grad/Reshape1^gradients/huber_loss/Mul_3_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/huber_loss/Mul_3_grad/Reshape*
_output_shapes

:  

:gradients/huber_loss/Mul_3_grad/tuple/control_dependency_1Identity)gradients/huber_loss/Mul_3_grad/Reshape_11^gradients/huber_loss/Mul_3_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/huber_loss/Mul_3_grad/Reshape_1*
_output_shapes
: 
q
.gradients/huber_loss/Add_grad/tuple/group_depsNoOp9^gradients/huber_loss/Mul_3_grad/tuple/control_dependency

6gradients/huber_loss/Add_grad/tuple/control_dependencyIdentity8gradients/huber_loss/Mul_3_grad/tuple/control_dependency/^gradients/huber_loss/Add_grad/tuple/group_deps*
_output_shapes

:  *
T0*:
_class0
.,loc:@gradients/huber_loss/Mul_3_grad/Reshape

8gradients/huber_loss/Add_grad/tuple/control_dependency_1Identity8gradients/huber_loss/Mul_3_grad/tuple/control_dependency/^gradients/huber_loss/Add_grad/tuple/group_deps*:
_class0
.,loc:@gradients/huber_loss/Mul_3_grad/Reshape*
_output_shapes

:  *
T0
h
%gradients/huber_loss/Mul_1_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
x
'gradients/huber_loss/Mul_1_grad/Shape_1Const*
valueB"        *
dtype0*
_output_shapes
:
л
5gradients/huber_loss/Mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs%gradients/huber_loss/Mul_1_grad/Shape'gradients/huber_loss/Mul_1_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0

#gradients/huber_loss/Mul_1_grad/MulMul6gradients/huber_loss/Add_grad/tuple/control_dependencyhuber_loss/Mul*
T0*
_output_shapes

:  
Ц
#gradients/huber_loss/Mul_1_grad/SumSum#gradients/huber_loss/Mul_1_grad/Mul5gradients/huber_loss/Mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
­
'gradients/huber_loss/Mul_1_grad/ReshapeReshape#gradients/huber_loss/Mul_1_grad/Sum%gradients/huber_loss/Mul_1_grad/Shape*
T0*
Tshape0*
_output_shapes
: 

%gradients/huber_loss/Mul_1_grad/Mul_1Mulhuber_loss/Const6gradients/huber_loss/Add_grad/tuple/control_dependency*
T0*
_output_shapes

:  
Ь
%gradients/huber_loss/Mul_1_grad/Sum_1Sum%gradients/huber_loss/Mul_1_grad/Mul_17gradients/huber_loss/Mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Л
)gradients/huber_loss/Mul_1_grad/Reshape_1Reshape%gradients/huber_loss/Mul_1_grad/Sum_1'gradients/huber_loss/Mul_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:  

0gradients/huber_loss/Mul_1_grad/tuple/group_depsNoOp(^gradients/huber_loss/Mul_1_grad/Reshape*^gradients/huber_loss/Mul_1_grad/Reshape_1
§
8gradients/huber_loss/Mul_1_grad/tuple/control_dependencyIdentity'gradients/huber_loss/Mul_1_grad/Reshape1^gradients/huber_loss/Mul_1_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/huber_loss/Mul_1_grad/Reshape*
_output_shapes
: 

:gradients/huber_loss/Mul_1_grad/tuple/control_dependency_1Identity)gradients/huber_loss/Mul_1_grad/Reshape_11^gradients/huber_loss/Mul_1_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/huber_loss/Mul_1_grad/Reshape_1*
_output_shapes

:  
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
valueB"        
л
5gradients/huber_loss/Mul_2_grad/BroadcastGradientArgsBroadcastGradientArgs%gradients/huber_loss/Mul_2_grad/Shape'gradients/huber_loss/Mul_2_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0

#gradients/huber_loss/Mul_2_grad/MulMul8gradients/huber_loss/Add_grad/tuple/control_dependency_1huber_loss/Sub_1*
_output_shapes

:  *
T0
Ц
#gradients/huber_loss/Mul_2_grad/SumSum#gradients/huber_loss/Mul_2_grad/Mul5gradients/huber_loss/Mul_2_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
­
'gradients/huber_loss/Mul_2_grad/ReshapeReshape#gradients/huber_loss/Mul_2_grad/Sum%gradients/huber_loss/Mul_2_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
Ѓ
%gradients/huber_loss/Mul_2_grad/Mul_1Mulhuber_loss/Mul_2/x8gradients/huber_loss/Add_grad/tuple/control_dependency_1*
_output_shapes

:  *
T0
Ь
%gradients/huber_loss/Mul_2_grad/Sum_1Sum%gradients/huber_loss/Mul_2_grad/Mul_17gradients/huber_loss/Mul_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Л
)gradients/huber_loss/Mul_2_grad/Reshape_1Reshape%gradients/huber_loss/Mul_2_grad/Sum_1'gradients/huber_loss/Mul_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:  

0gradients/huber_loss/Mul_2_grad/tuple/group_depsNoOp(^gradients/huber_loss/Mul_2_grad/Reshape*^gradients/huber_loss/Mul_2_grad/Reshape_1
§
8gradients/huber_loss/Mul_2_grad/tuple/control_dependencyIdentity'gradients/huber_loss/Mul_2_grad/Reshape1^gradients/huber_loss/Mul_2_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/huber_loss/Mul_2_grad/Reshape*
_output_shapes
: 

:gradients/huber_loss/Mul_2_grad/tuple/control_dependency_1Identity)gradients/huber_loss/Mul_2_grad/Reshape_11^gradients/huber_loss/Mul_2_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/huber_loss/Mul_2_grad/Reshape_1*
_output_shapes

:  
Ё
!gradients/huber_loss/Mul_grad/MulMul:gradients/huber_loss/Mul_1_grad/tuple/control_dependency_1huber_loss/Minimum*
_output_shapes

:  *
T0
Ѓ
#gradients/huber_loss/Mul_grad/Mul_1Mul:gradients/huber_loss/Mul_1_grad/tuple/control_dependency_1huber_loss/Minimum*
_output_shapes

:  *
T0

.gradients/huber_loss/Mul_grad/tuple/group_depsNoOp"^gradients/huber_loss/Mul_grad/Mul$^gradients/huber_loss/Mul_grad/Mul_1
ѕ
6gradients/huber_loss/Mul_grad/tuple/control_dependencyIdentity!gradients/huber_loss/Mul_grad/Mul/^gradients/huber_loss/Mul_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients/huber_loss/Mul_grad/Mul*
_output_shapes

:  
ћ
8gradients/huber_loss/Mul_grad/tuple/control_dependency_1Identity#gradients/huber_loss/Mul_grad/Mul_1/^gradients/huber_loss/Mul_grad/tuple/group_deps*6
_class,
*(loc:@gradients/huber_loss/Mul_grad/Mul_1*
_output_shapes

:  *
T0

#gradients/huber_loss/Sub_1_grad/NegNeg:gradients/huber_loss/Mul_2_grad/tuple/control_dependency_1*
_output_shapes

:  *
T0

0gradients/huber_loss/Sub_1_grad/tuple/group_depsNoOp;^gradients/huber_loss/Mul_2_grad/tuple/control_dependency_1$^gradients/huber_loss/Sub_1_grad/Neg

8gradients/huber_loss/Sub_1_grad/tuple/control_dependencyIdentity:gradients/huber_loss/Mul_2_grad/tuple/control_dependency_11^gradients/huber_loss/Sub_1_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/huber_loss/Mul_2_grad/Reshape_1*
_output_shapes

:  
џ
:gradients/huber_loss/Sub_1_grad/tuple/control_dependency_1Identity#gradients/huber_loss/Sub_1_grad/Neg1^gradients/huber_loss/Sub_1_grad/tuple/group_deps*
_output_shapes

:  *
T0*6
_class,
*(loc:@gradients/huber_loss/Sub_1_grad/Neg
Ў
gradients/AddN_1AddN6gradients/huber_loss/Mul_grad/tuple/control_dependency8gradients/huber_loss/Mul_grad/tuple/control_dependency_1:gradients/huber_loss/Sub_1_grad/tuple/control_dependency_1*
T0*4
_class*
(&loc:@gradients/huber_loss/Mul_grad/Mul*
N*
_output_shapes

:  
x
'gradients/huber_loss/Minimum_grad/ShapeConst*
valueB"        *
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
valueB"        *
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
Ф
'gradients/huber_loss/Minimum_grad/zerosFill)gradients/huber_loss/Minimum_grad/Shape_2-gradients/huber_loss/Minimum_grad/zeros/Const*
_output_shapes

:  *
T0*

index_type0

+gradients/huber_loss/Minimum_grad/LessEqual	LessEqualhuber_loss/Abshuber_loss/Minimum/y*
T0*
_output_shapes

:  
с
7gradients/huber_loss/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgs'gradients/huber_loss/Minimum_grad/Shape)gradients/huber_loss/Minimum_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
У
(gradients/huber_loss/Minimum_grad/SelectSelect+gradients/huber_loss/Minimum_grad/LessEqualgradients/AddN_1'gradients/huber_loss/Minimum_grad/zeros*
_output_shapes

:  *
T0
Х
*gradients/huber_loss/Minimum_grad/Select_1Select+gradients/huber_loss/Minimum_grad/LessEqual'gradients/huber_loss/Minimum_grad/zerosgradients/AddN_1*
_output_shapes

:  *
T0
Я
%gradients/huber_loss/Minimum_grad/SumSum(gradients/huber_loss/Minimum_grad/Select7gradients/huber_loss/Minimum_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Л
)gradients/huber_loss/Minimum_grad/ReshapeReshape%gradients/huber_loss/Minimum_grad/Sum'gradients/huber_loss/Minimum_grad/Shape*
T0*
Tshape0*
_output_shapes

:  
е
'gradients/huber_loss/Minimum_grad/Sum_1Sum*gradients/huber_loss/Minimum_grad/Select_19gradients/huber_loss/Minimum_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Й
+gradients/huber_loss/Minimum_grad/Reshape_1Reshape'gradients/huber_loss/Minimum_grad/Sum_1)gradients/huber_loss/Minimum_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 

2gradients/huber_loss/Minimum_grad/tuple/group_depsNoOp*^gradients/huber_loss/Minimum_grad/Reshape,^gradients/huber_loss/Minimum_grad/Reshape_1

:gradients/huber_loss/Minimum_grad/tuple/control_dependencyIdentity)gradients/huber_loss/Minimum_grad/Reshape3^gradients/huber_loss/Minimum_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/huber_loss/Minimum_grad/Reshape*
_output_shapes

:  

<gradients/huber_loss/Minimum_grad/tuple/control_dependency_1Identity+gradients/huber_loss/Minimum_grad/Reshape_13^gradients/huber_loss/Minimum_grad/tuple/group_deps*
_output_shapes
: *
T0*>
_class4
20loc:@gradients/huber_loss/Minimum_grad/Reshape_1
ў
gradients/AddN_2AddN8gradients/huber_loss/Sub_1_grad/tuple/control_dependency:gradients/huber_loss/Minimum_grad/tuple/control_dependency*
_output_shapes

:  *
T0*<
_class2
0.loc:@gradients/huber_loss/Mul_2_grad/Reshape_1*
N
c
"gradients/huber_loss/Abs_grad/SignSignhuber_loss/Sub*
T0*
_output_shapes

:  

!gradients/huber_loss/Abs_grad/mulMulgradients/AddN_2"gradients/huber_loss/Abs_grad/Sign*
T0*
_output_shapes

:  
f
#gradients/huber_loss/Sub_grad/ShapeShapeSum*
_output_shapes
:*
T0*
out_type0
r
%gradients/huber_loss/Sub_grad/Shape_1ShapePlaceholder_2*
_output_shapes
:*
T0*
out_type0
е
3gradients/huber_loss/Sub_grad/BroadcastGradientArgsBroadcastGradientArgs#gradients/huber_loss/Sub_grad/Shape%gradients/huber_loss/Sub_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Р
!gradients/huber_loss/Sub_grad/SumSum!gradients/huber_loss/Abs_grad/mul3gradients/huber_loss/Sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
И
%gradients/huber_loss/Sub_grad/ReshapeReshape!gradients/huber_loss/Sub_grad/Sum#gradients/huber_loss/Sub_grad/Shape*'
_output_shapes
: џџџџџџџџџ*
T0*
Tshape0
Ф
#gradients/huber_loss/Sub_grad/Sum_1Sum!gradients/huber_loss/Abs_grad/mul5gradients/huber_loss/Sub_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
p
!gradients/huber_loss/Sub_grad/NegNeg#gradients/huber_loss/Sub_grad/Sum_1*
_output_shapes
:*
T0
М
'gradients/huber_loss/Sub_grad/Reshape_1Reshape!gradients/huber_loss/Sub_grad/Neg%gradients/huber_loss/Sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ 

.gradients/huber_loss/Sub_grad/tuple/group_depsNoOp&^gradients/huber_loss/Sub_grad/Reshape(^gradients/huber_loss/Sub_grad/Reshape_1

6gradients/huber_loss/Sub_grad/tuple/control_dependencyIdentity%gradients/huber_loss/Sub_grad/Reshape/^gradients/huber_loss/Sub_grad/tuple/group_deps*
T0*8
_class.
,*loc:@gradients/huber_loss/Sub_grad/Reshape*'
_output_shapes
: џџџџџџџџџ

8gradients/huber_loss/Sub_grad/tuple/control_dependency_1Identity'gradients/huber_loss/Sub_grad/Reshape_1/^gradients/huber_loss/Sub_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/huber_loss/Sub_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ 
[
gradients/Sum_grad/ShapeShapemul*
T0*
out_type0*
_output_shapes
:

gradients/Sum_grad/SizeConst*+
_class!
loc:@gradients/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 

gradients/Sum_grad/addAddSum/reduction_indicesgradients/Sum_grad/Size*
_output_shapes
: *
T0*+
_class!
loc:@gradients/Sum_grad/Shape
Ё
gradients/Sum_grad/modFloorModgradients/Sum_grad/addgradients/Sum_grad/Size*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
: 

gradients/Sum_grad/Shape_1Const*+
_class!
loc:@gradients/Sum_grad/Shape*
valueB *
dtype0*
_output_shapes
: 

gradients/Sum_grad/range/startConst*+
_class!
loc:@gradients/Sum_grad/Shape*
value	B : *
dtype0*
_output_shapes
: 

gradients/Sum_grad/range/deltaConst*+
_class!
loc:@gradients/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
Я
gradients/Sum_grad/rangeRangegradients/Sum_grad/range/startgradients/Sum_grad/Sizegradients/Sum_grad/range/delta*
_output_shapes
:*

Tidx0*+
_class!
loc:@gradients/Sum_grad/Shape

gradients/Sum_grad/Fill/valueConst*+
_class!
loc:@gradients/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
К
gradients/Sum_grad/FillFillgradients/Sum_grad/Shape_1gradients/Sum_grad/Fill/value*+
_class!
loc:@gradients/Sum_grad/Shape*

index_type0*
_output_shapes
: *
T0
њ
 gradients/Sum_grad/DynamicStitchDynamicStitchgradients/Sum_grad/rangegradients/Sum_grad/modgradients/Sum_grad/Shapegradients/Sum_grad/Fill*#
_output_shapes
:џџџџџџџџџ*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*
N

gradients/Sum_grad/Maximum/yConst*+
_class!
loc:@gradients/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
Р
gradients/Sum_grad/MaximumMaximum gradients/Sum_grad/DynamicStitchgradients/Sum_grad/Maximum/y*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*#
_output_shapes
:џџџџџџџџџ
Џ
gradients/Sum_grad/floordivFloorDivgradients/Sum_grad/Shapegradients/Sum_grad/Maximum*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*
_output_shapes
:
А
gradients/Sum_grad/ReshapeReshape6gradients/huber_loss/Sub_grad/tuple/control_dependency gradients/Sum_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
 
gradients/Sum_grad/TileTilegradients/Sum_grad/Reshapegradients/Sum_grad/floordiv*

Tmultiples0*
T0*+
_output_shapes
: џџџџџџџџџ
f
gradients/mul_grad/ShapeShapemain/transpose*
T0*
out_type0*
_output_shapes
:
d
gradients/mul_grad/Shape_1Shape
ExpandDims*
T0*
out_type0*
_output_shapes
:
Д
(gradients/mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_grad/Shapegradients/mul_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
x
gradients/mul_grad/MulMulgradients/Sum_grad/Tile
ExpandDims*
T0*+
_output_shapes
: џџџџџџџџџ

gradients/mul_grad/SumSumgradients/mul_grad/Mul(gradients/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

gradients/mul_grad/ReshapeReshapegradients/mul_grad/Sumgradients/mul_grad/Shape*
T0*
Tshape0*+
_output_shapes
: џџџџџџџџџ
~
gradients/mul_grad/Mul_1Mulmain/transposegradients/Sum_grad/Tile*
T0*+
_output_shapes
: џџџџџџџџџ
Ѕ
gradients/mul_grad/Sum_1Sumgradients/mul_grad/Mul_1*gradients/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ё
gradients/mul_grad/Reshape_1Reshapegradients/mul_grad/Sum_1gradients/mul_grad/Shape_1*+
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
g
#gradients/mul_grad/tuple/group_depsNoOp^gradients/mul_grad/Reshape^gradients/mul_grad/Reshape_1
о
+gradients/mul_grad/tuple/control_dependencyIdentitygradients/mul_grad/Reshape$^gradients/mul_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/mul_grad/Reshape*+
_output_shapes
: џџџџџџџџџ
ф
-gradients/mul_grad/tuple/control_dependency_1Identitygradients/mul_grad/Reshape_1$^gradients/mul_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/mul_grad/Reshape_1*+
_output_shapes
:џџџџџџџџџ
~
/gradients/main/transpose_grad/InvertPermutationInvertPermutationmain/transpose/perm*
_output_shapes
:*
T0
е
'gradients/main/transpose_grad/transpose	Transpose+gradients/mul_grad/tuple/control_dependency/gradients/main/transpose_grad/InvertPermutation*
Tperm0*
T0*+
_output_shapes
: џџџџџџџџџ
ѓ
'gradients/main/transpose/x_grad/unstackUnpack'gradients/main/transpose_grad/transpose*і
_output_shapesу
р:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*	
num *
T0*

axis 
b
0gradients/main/transpose/x_grad/tuple/group_depsNoOp(^gradients/main/transpose/x_grad/unstack

8gradients/main/transpose/x_grad/tuple/control_dependencyIdentity'gradients/main/transpose/x_grad/unstack1^gradients/main/transpose/x_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack

:gradients/main/transpose/x_grad/tuple/control_dependency_1Identity)gradients/main/transpose/x_grad/unstack:11^gradients/main/transpose/x_grad/tuple/group_deps*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:џџџџџџџџџ*
T0

:gradients/main/transpose/x_grad/tuple/control_dependency_2Identity)gradients/main/transpose/x_grad/unstack:21^gradients/main/transpose/x_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack

:gradients/main/transpose/x_grad/tuple/control_dependency_3Identity)gradients/main/transpose/x_grad/unstack:31^gradients/main/transpose/x_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack

:gradients/main/transpose/x_grad/tuple/control_dependency_4Identity)gradients/main/transpose/x_grad/unstack:41^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:џџџџџџџџџ

:gradients/main/transpose/x_grad/tuple/control_dependency_5Identity)gradients/main/transpose/x_grad/unstack:51^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:џџџџџџџџџ

:gradients/main/transpose/x_grad/tuple/control_dependency_6Identity)gradients/main/transpose/x_grad/unstack:61^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:џџџџџџџџџ

:gradients/main/transpose/x_grad/tuple/control_dependency_7Identity)gradients/main/transpose/x_grad/unstack:71^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:џџџџџџџџџ

:gradients/main/transpose/x_grad/tuple/control_dependency_8Identity)gradients/main/transpose/x_grad/unstack:81^gradients/main/transpose/x_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack

:gradients/main/transpose/x_grad/tuple/control_dependency_9Identity)gradients/main/transpose/x_grad/unstack:91^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:џџџџџџџџџ

;gradients/main/transpose/x_grad/tuple/control_dependency_10Identity*gradients/main/transpose/x_grad/unstack:101^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:џџџџџџџџџ

;gradients/main/transpose/x_grad/tuple/control_dependency_11Identity*gradients/main/transpose/x_grad/unstack:111^gradients/main/transpose/x_grad/tuple/group_deps*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:џџџџџџџџџ*
T0

;gradients/main/transpose/x_grad/tuple/control_dependency_12Identity*gradients/main/transpose/x_grad/unstack:121^gradients/main/transpose/x_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack

;gradients/main/transpose/x_grad/tuple/control_dependency_13Identity*gradients/main/transpose/x_grad/unstack:131^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:џџџџџџџџџ

;gradients/main/transpose/x_grad/tuple/control_dependency_14Identity*gradients/main/transpose/x_grad/unstack:141^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:џџџџџџџџџ

;gradients/main/transpose/x_grad/tuple/control_dependency_15Identity*gradients/main/transpose/x_grad/unstack:151^gradients/main/transpose/x_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack

;gradients/main/transpose/x_grad/tuple/control_dependency_16Identity*gradients/main/transpose/x_grad/unstack:161^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:џџџџџџџџџ

;gradients/main/transpose/x_grad/tuple/control_dependency_17Identity*gradients/main/transpose/x_grad/unstack:171^gradients/main/transpose/x_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack

;gradients/main/transpose/x_grad/tuple/control_dependency_18Identity*gradients/main/transpose/x_grad/unstack:181^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:џџџџџџџџџ

;gradients/main/transpose/x_grad/tuple/control_dependency_19Identity*gradients/main/transpose/x_grad/unstack:191^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:џџџџџџџџџ

;gradients/main/transpose/x_grad/tuple/control_dependency_20Identity*gradients/main/transpose/x_grad/unstack:201^gradients/main/transpose/x_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack

;gradients/main/transpose/x_grad/tuple/control_dependency_21Identity*gradients/main/transpose/x_grad/unstack:211^gradients/main/transpose/x_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack

;gradients/main/transpose/x_grad/tuple/control_dependency_22Identity*gradients/main/transpose/x_grad/unstack:221^gradients/main/transpose/x_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack

;gradients/main/transpose/x_grad/tuple/control_dependency_23Identity*gradients/main/transpose/x_grad/unstack:231^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:џџџџџџџџџ

;gradients/main/transpose/x_grad/tuple/control_dependency_24Identity*gradients/main/transpose/x_grad/unstack:241^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:џџџџџџџџџ

;gradients/main/transpose/x_grad/tuple/control_dependency_25Identity*gradients/main/transpose/x_grad/unstack:251^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:џџџџџџџџџ

;gradients/main/transpose/x_grad/tuple/control_dependency_26Identity*gradients/main/transpose/x_grad/unstack:261^gradients/main/transpose/x_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack

;gradients/main/transpose/x_grad/tuple/control_dependency_27Identity*gradients/main/transpose/x_grad/unstack:271^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:џџџџџџџџџ

;gradients/main/transpose/x_grad/tuple/control_dependency_28Identity*gradients/main/transpose/x_grad/unstack:281^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:џџџџџџџџџ

;gradients/main/transpose/x_grad/tuple/control_dependency_29Identity*gradients/main/transpose/x_grad/unstack:291^gradients/main/transpose/x_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack

;gradients/main/transpose/x_grad/tuple/control_dependency_30Identity*gradients/main/transpose/x_grad/unstack:301^gradients/main/transpose/x_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack*'
_output_shapes
:џџџџџџџџџ

;gradients/main/transpose/x_grad/tuple/control_dependency_31Identity*gradients/main/transpose/x_grad/unstack:311^gradients/main/transpose/x_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*:
_class0
.,loc:@gradients/main/transpose/x_grad/unstack

 gradients/main/split_grad/concatConcatV28gradients/main/transpose/x_grad/tuple/control_dependency:gradients/main/transpose/x_grad/tuple/control_dependency_1:gradients/main/transpose/x_grad/tuple/control_dependency_2:gradients/main/transpose/x_grad/tuple/control_dependency_3:gradients/main/transpose/x_grad/tuple/control_dependency_4:gradients/main/transpose/x_grad/tuple/control_dependency_5:gradients/main/transpose/x_grad/tuple/control_dependency_6:gradients/main/transpose/x_grad/tuple/control_dependency_7:gradients/main/transpose/x_grad/tuple/control_dependency_8:gradients/main/transpose/x_grad/tuple/control_dependency_9;gradients/main/transpose/x_grad/tuple/control_dependency_10;gradients/main/transpose/x_grad/tuple/control_dependency_11;gradients/main/transpose/x_grad/tuple/control_dependency_12;gradients/main/transpose/x_grad/tuple/control_dependency_13;gradients/main/transpose/x_grad/tuple/control_dependency_14;gradients/main/transpose/x_grad/tuple/control_dependency_15;gradients/main/transpose/x_grad/tuple/control_dependency_16;gradients/main/transpose/x_grad/tuple/control_dependency_17;gradients/main/transpose/x_grad/tuple/control_dependency_18;gradients/main/transpose/x_grad/tuple/control_dependency_19;gradients/main/transpose/x_grad/tuple/control_dependency_20;gradients/main/transpose/x_grad/tuple/control_dependency_21;gradients/main/transpose/x_grad/tuple/control_dependency_22;gradients/main/transpose/x_grad/tuple/control_dependency_23;gradients/main/transpose/x_grad/tuple/control_dependency_24;gradients/main/transpose/x_grad/tuple/control_dependency_25;gradients/main/transpose/x_grad/tuple/control_dependency_26;gradients/main/transpose/x_grad/tuple/control_dependency_27;gradients/main/transpose/x_grad/tuple/control_dependency_28;gradients/main/transpose/x_grad/tuple/control_dependency_29;gradients/main/transpose/x_grad/tuple/control_dependency_30;gradients/main/transpose/x_grad/tuple/control_dependency_31main/split/split_dim*

Tidx0*
T0*
N *'
_output_shapes
:џџџџџџџџџ

/gradients/main/dense_4/BiasAdd_grad/BiasAddGradBiasAddGrad gradients/main/split_grad/concat*
data_formatNHWC*
_output_shapes
:*
T0

4gradients/main/dense_4/BiasAdd_grad/tuple/group_depsNoOp0^gradients/main/dense_4/BiasAdd_grad/BiasAddGrad!^gradients/main/split_grad/concat

<gradients/main/dense_4/BiasAdd_grad/tuple/control_dependencyIdentity gradients/main/split_grad/concat5^gradients/main/dense_4/BiasAdd_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/main/split_grad/concat*'
_output_shapes
:џџџџџџџџџ

>gradients/main/dense_4/BiasAdd_grad/tuple/control_dependency_1Identity/gradients/main/dense_4/BiasAdd_grad/BiasAddGrad5^gradients/main/dense_4/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*B
_class8
64loc:@gradients/main/dense_4/BiasAdd_grad/BiasAddGrad
ф
)gradients/main/dense_4/MatMul_grad/MatMulMatMul<gradients/main/dense_4/BiasAdd_grad/tuple/control_dependencymain/dense_4/kernel/read*
transpose_b(*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( 
ж
+gradients/main/dense_4/MatMul_grad/MatMul_1MatMulmain/dense_3/Relu<gradients/main/dense_4/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	*
transpose_a(*
transpose_b( *
T0

3gradients/main/dense_4/MatMul_grad/tuple/group_depsNoOp*^gradients/main/dense_4/MatMul_grad/MatMul,^gradients/main/dense_4/MatMul_grad/MatMul_1

;gradients/main/dense_4/MatMul_grad/tuple/control_dependencyIdentity)gradients/main/dense_4/MatMul_grad/MatMul4^gradients/main/dense_4/MatMul_grad/tuple/group_deps*<
_class2
0.loc:@gradients/main/dense_4/MatMul_grad/MatMul*(
_output_shapes
:џџџџџџџџџ*
T0

=gradients/main/dense_4/MatMul_grad/tuple/control_dependency_1Identity+gradients/main/dense_4/MatMul_grad/MatMul_14^gradients/main/dense_4/MatMul_grad/tuple/group_deps*
_output_shapes
:	*
T0*>
_class4
20loc:@gradients/main/dense_4/MatMul_grad/MatMul_1
И
)gradients/main/dense_3/Relu_grad/ReluGradReluGrad;gradients/main/dense_4/MatMul_grad/tuple/control_dependencymain/dense_3/Relu*
T0*(
_output_shapes
:џџџџџџџџџ
І
/gradients/main/dense_3/BiasAdd_grad/BiasAddGradBiasAddGrad)gradients/main/dense_3/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:

4gradients/main/dense_3/BiasAdd_grad/tuple/group_depsNoOp0^gradients/main/dense_3/BiasAdd_grad/BiasAddGrad*^gradients/main/dense_3/Relu_grad/ReluGrad

<gradients/main/dense_3/BiasAdd_grad/tuple/control_dependencyIdentity)gradients/main/dense_3/Relu_grad/ReluGrad5^gradients/main/dense_3/BiasAdd_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/main/dense_3/Relu_grad/ReluGrad*(
_output_shapes
:џџџџџџџџџ

>gradients/main/dense_3/BiasAdd_grad/tuple/control_dependency_1Identity/gradients/main/dense_3/BiasAdd_grad/BiasAddGrad5^gradients/main/dense_3/BiasAdd_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/main/dense_3/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
ф
)gradients/main/dense_3/MatMul_grad/MatMulMatMul<gradients/main/dense_3/BiasAdd_grad/tuple/control_dependencymain/dense_3/kernel/read*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(*
T0
з
+gradients/main/dense_3/MatMul_grad/MatMul_1MatMulmain/dense_2/Relu<gradients/main/dense_3/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
*
transpose_a(*
transpose_b( 

3gradients/main/dense_3/MatMul_grad/tuple/group_depsNoOp*^gradients/main/dense_3/MatMul_grad/MatMul,^gradients/main/dense_3/MatMul_grad/MatMul_1

;gradients/main/dense_3/MatMul_grad/tuple/control_dependencyIdentity)gradients/main/dense_3/MatMul_grad/MatMul4^gradients/main/dense_3/MatMul_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/main/dense_3/MatMul_grad/MatMul*(
_output_shapes
:џџџџџџџџџ

=gradients/main/dense_3/MatMul_grad/tuple/control_dependency_1Identity+gradients/main/dense_3/MatMul_grad/MatMul_14^gradients/main/dense_3/MatMul_grad/tuple/group_deps*>
_class4
20loc:@gradients/main/dense_3/MatMul_grad/MatMul_1* 
_output_shapes
:
*
T0
И
)gradients/main/dense_2/Relu_grad/ReluGradReluGrad;gradients/main/dense_3/MatMul_grad/tuple/control_dependencymain/dense_2/Relu*(
_output_shapes
:џџџџџџџџџ*
T0
І
/gradients/main/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad)gradients/main/dense_2/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes	
:*
T0

4gradients/main/dense_2/BiasAdd_grad/tuple/group_depsNoOp0^gradients/main/dense_2/BiasAdd_grad/BiasAddGrad*^gradients/main/dense_2/Relu_grad/ReluGrad

<gradients/main/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity)gradients/main/dense_2/Relu_grad/ReluGrad5^gradients/main/dense_2/BiasAdd_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/main/dense_2/Relu_grad/ReluGrad*(
_output_shapes
:џџџџџџџџџ

>gradients/main/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity/gradients/main/dense_2/BiasAdd_grad/BiasAddGrad5^gradients/main/dense_2/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:*
T0*B
_class8
64loc:@gradients/main/dense_2/BiasAdd_grad/BiasAddGrad
ф
)gradients/main/dense_2/MatMul_grad/MatMulMatMul<gradients/main/dense_2/BiasAdd_grad/tuple/control_dependencymain/dense_2/kernel/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(
Ю
+gradients/main/dense_2/MatMul_grad/MatMul_1MatMulmain/Mul<gradients/main/dense_2/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
*
transpose_a(*
transpose_b( 

3gradients/main/dense_2/MatMul_grad/tuple/group_depsNoOp*^gradients/main/dense_2/MatMul_grad/MatMul,^gradients/main/dense_2/MatMul_grad/MatMul_1

;gradients/main/dense_2/MatMul_grad/tuple/control_dependencyIdentity)gradients/main/dense_2/MatMul_grad/MatMul4^gradients/main/dense_2/MatMul_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/main/dense_2/MatMul_grad/MatMul*(
_output_shapes
:џџџџџџџџџ

=gradients/main/dense_2/MatMul_grad/tuple/control_dependency_1Identity+gradients/main/dense_2/MatMul_grad/MatMul_14^gradients/main/dense_2/MatMul_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/main/dense_2/MatMul_grad/MatMul_1* 
_output_shapes
:

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
У
-gradients/main/Mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/main/Mul_grad/Shapegradients/main/Mul_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ѕ
gradients/main/Mul_grad/MulMul;gradients/main/dense_2/MatMul_grad/tuple/control_dependencymain/dense_1/Relu*(
_output_shapes
:џџџџџџџџџ*
T0
Ў
gradients/main/Mul_grad/SumSumgradients/main/Mul_grad/Mul-gradients/main/Mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ї
gradients/main/Mul_grad/ReshapeReshapegradients/main/Mul_grad/Sumgradients/main/Mul_grad/Shape*(
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
Ѕ
gradients/main/Mul_grad/Mul_1Mulmain/dense/Selu;gradients/main/dense_2/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ
Д
gradients/main/Mul_grad/Sum_1Sumgradients/main/Mul_grad/Mul_1/gradients/main/Mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
­
!gradients/main/Mul_grad/Reshape_1Reshapegradients/main/Mul_grad/Sum_1gradients/main/Mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
v
(gradients/main/Mul_grad/tuple/group_depsNoOp ^gradients/main/Mul_grad/Reshape"^gradients/main/Mul_grad/Reshape_1
я
0gradients/main/Mul_grad/tuple/control_dependencyIdentitygradients/main/Mul_grad/Reshape)^gradients/main/Mul_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/main/Mul_grad/Reshape*(
_output_shapes
:џџџџџџџџџ
ѕ
2gradients/main/Mul_grad/tuple/control_dependency_1Identity!gradients/main/Mul_grad/Reshape_1)^gradients/main/Mul_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*4
_class*
(&loc:@gradients/main/Mul_grad/Reshape_1
Љ
'gradients/main/dense/Selu_grad/SeluGradSeluGrad0gradients/main/Mul_grad/tuple/control_dependencymain/dense/Selu*
T0*(
_output_shapes
:џџџџџџџџџ
Џ
)gradients/main/dense_1/Relu_grad/ReluGradReluGrad2gradients/main/Mul_grad/tuple/control_dependency_1main/dense_1/Relu*
T0*(
_output_shapes
:џџџџџџџџџ
Ђ
-gradients/main/dense/BiasAdd_grad/BiasAddGradBiasAddGrad'gradients/main/dense/Selu_grad/SeluGrad*
T0*
data_formatNHWC*
_output_shapes	
:

2gradients/main/dense/BiasAdd_grad/tuple/group_depsNoOp.^gradients/main/dense/BiasAdd_grad/BiasAddGrad(^gradients/main/dense/Selu_grad/SeluGrad

:gradients/main/dense/BiasAdd_grad/tuple/control_dependencyIdentity'gradients/main/dense/Selu_grad/SeluGrad3^gradients/main/dense/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*:
_class0
.,loc:@gradients/main/dense/Selu_grad/SeluGrad

<gradients/main/dense/BiasAdd_grad/tuple/control_dependency_1Identity-gradients/main/dense/BiasAdd_grad/BiasAddGrad3^gradients/main/dense/BiasAdd_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/main/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
І
/gradients/main/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad)gradients/main/dense_1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:

4gradients/main/dense_1/BiasAdd_grad/tuple/group_depsNoOp0^gradients/main/dense_1/BiasAdd_grad/BiasAddGrad*^gradients/main/dense_1/Relu_grad/ReluGrad

<gradients/main/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity)gradients/main/dense_1/Relu_grad/ReluGrad5^gradients/main/dense_1/BiasAdd_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/main/dense_1/Relu_grad/ReluGrad*(
_output_shapes
:џџџџџџџџџ

>gradients/main/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity/gradients/main/dense_1/BiasAdd_grad/BiasAddGrad5^gradients/main/dense_1/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:*
T0*B
_class8
64loc:@gradients/main/dense_1/BiasAdd_grad/BiasAddGrad
н
'gradients/main/dense/MatMul_grad/MatMulMatMul:gradients/main/dense/BiasAdd_grad/tuple/control_dependencymain/dense/kernel/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(
Э
)gradients/main/dense/MatMul_grad/MatMul_1MatMulmain/Reshape:gradients/main/dense/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	*
transpose_a(*
transpose_b( *
T0

1gradients/main/dense/MatMul_grad/tuple/group_depsNoOp(^gradients/main/dense/MatMul_grad/MatMul*^gradients/main/dense/MatMul_grad/MatMul_1

9gradients/main/dense/MatMul_grad/tuple/control_dependencyIdentity'gradients/main/dense/MatMul_grad/MatMul2^gradients/main/dense/MatMul_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/main/dense/MatMul_grad/MatMul*'
_output_shapes
:џџџџџџџџџ

;gradients/main/dense/MatMul_grad/tuple/control_dependency_1Identity)gradients/main/dense/MatMul_grad/MatMul_12^gradients/main/dense/MatMul_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/main/dense/MatMul_grad/MatMul_1*
_output_shapes
:	
у
)gradients/main/dense_1/MatMul_grad/MatMulMatMul<gradients/main/dense_1/BiasAdd_grad/tuple/control_dependencymain/dense_1/kernel/read*
T0*'
_output_shapes
:џџџџџџџџџ@*
transpose_a( *
transpose_b(
Э
+gradients/main/dense_1/MatMul_grad/MatMul_1MatMulmain/Cos<gradients/main/dense_1/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	@*
transpose_a(*
transpose_b( 

3gradients/main/dense_1/MatMul_grad/tuple/group_depsNoOp*^gradients/main/dense_1/MatMul_grad/MatMul,^gradients/main/dense_1/MatMul_grad/MatMul_1

;gradients/main/dense_1/MatMul_grad/tuple/control_dependencyIdentity)gradients/main/dense_1/MatMul_grad/MatMul4^gradients/main/dense_1/MatMul_grad/tuple/group_deps*<
_class2
0.loc:@gradients/main/dense_1/MatMul_grad/MatMul*'
_output_shapes
:џџџџџџџџџ@*
T0

=gradients/main/dense_1/MatMul_grad/tuple/control_dependency_1Identity+gradients/main/dense_1/MatMul_grad/MatMul_14^gradients/main/dense_1/MatMul_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/main/dense_1/MatMul_grad/MatMul_1*
_output_shapes
:	@

beta1_power/initial_valueConst*
dtype0*
_output_shapes
: *"
_class
loc:@main/dense/bias*
valueB
 *fff?

beta1_power
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *"
_class
loc:@main/dense/bias*
	container 
В
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
use_locking(*
T0*"
_class
loc:@main/dense/bias*
validate_shape(*
_output_shapes
: 
n
beta1_power/readIdentitybeta1_power*
T0*"
_class
loc:@main/dense/bias*
_output_shapes
: 

beta2_power/initial_valueConst*"
_class
loc:@main/dense/bias*
valueB
 *wО?*
dtype0*
_output_shapes
: 

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
В
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
Ѕ
(main/dense/kernel/Adam/Initializer/zerosConst*$
_class
loc:@main/dense/kernel*
valueB	*    *
dtype0*
_output_shapes
:	
В
main/dense/kernel/Adam
VariableV2*
dtype0*
_output_shapes
:	*
shared_name *$
_class
loc:@main/dense/kernel*
	container *
shape:	
т
main/dense/kernel/Adam/AssignAssignmain/dense/kernel/Adam(main/dense/kernel/Adam/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@main/dense/kernel*
validate_shape(*
_output_shapes
:	

main/dense/kernel/Adam/readIdentitymain/dense/kernel/Adam*
T0*$
_class
loc:@main/dense/kernel*
_output_shapes
:	
Ї
*main/dense/kernel/Adam_1/Initializer/zerosConst*$
_class
loc:@main/dense/kernel*
valueB	*    *
dtype0*
_output_shapes
:	
Д
main/dense/kernel/Adam_1
VariableV2*
shape:	*
dtype0*
_output_shapes
:	*
shared_name *$
_class
loc:@main/dense/kernel*
	container 
ш
main/dense/kernel/Adam_1/AssignAssignmain/dense/kernel/Adam_1*main/dense/kernel/Adam_1/Initializer/zeros*
T0*$
_class
loc:@main/dense/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(

main/dense/kernel/Adam_1/readIdentitymain/dense/kernel/Adam_1*
T0*$
_class
loc:@main/dense/kernel*
_output_shapes
:	

&main/dense/bias/Adam/Initializer/zerosConst*"
_class
loc:@main/dense/bias*
valueB*    *
dtype0*
_output_shapes	
:
І
main/dense/bias/Adam
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *"
_class
loc:@main/dense/bias*
	container 
ж
main/dense/bias/Adam/AssignAssignmain/dense/bias/Adam&main/dense/bias/Adam/Initializer/zeros*
T0*"
_class
loc:@main/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(

main/dense/bias/Adam/readIdentitymain/dense/bias/Adam*
T0*"
_class
loc:@main/dense/bias*
_output_shapes	
:

(main/dense/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes	
:*"
_class
loc:@main/dense/bias*
valueB*    
Ј
main/dense/bias/Adam_1
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *"
_class
loc:@main/dense/bias*
	container *
shape:
м
main/dense/bias/Adam_1/AssignAssignmain/dense/bias/Adam_1(main/dense/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@main/dense/bias*
validate_shape(*
_output_shapes	
:

main/dense/bias/Adam_1/readIdentitymain/dense/bias/Adam_1*
T0*"
_class
loc:@main/dense/bias*
_output_shapes	
:
Г
:main/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*&
_class
loc:@main/dense_1/kernel*
valueB"@      *
dtype0*
_output_shapes
:

0main/dense_1/kernel/Adam/Initializer/zeros/ConstConst*&
_class
loc:@main/dense_1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

*main/dense_1/kernel/Adam/Initializer/zerosFill:main/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensor0main/dense_1/kernel/Adam/Initializer/zeros/Const*
T0*&
_class
loc:@main/dense_1/kernel*

index_type0*
_output_shapes
:	@
Ж
main/dense_1/kernel/Adam
VariableV2*
shared_name *&
_class
loc:@main/dense_1/kernel*
	container *
shape:	@*
dtype0*
_output_shapes
:	@
ъ
main/dense_1/kernel/Adam/AssignAssignmain/dense_1/kernel/Adam*main/dense_1/kernel/Adam/Initializer/zeros*
_output_shapes
:	@*
use_locking(*
T0*&
_class
loc:@main/dense_1/kernel*
validate_shape(

main/dense_1/kernel/Adam/readIdentitymain/dense_1/kernel/Adam*
T0*&
_class
loc:@main/dense_1/kernel*
_output_shapes
:	@
Е
<main/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*&
_class
loc:@main/dense_1/kernel*
valueB"@      *
dtype0*
_output_shapes
:

2main/dense_1/kernel/Adam_1/Initializer/zeros/ConstConst*&
_class
loc:@main/dense_1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

,main/dense_1/kernel/Adam_1/Initializer/zerosFill<main/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor2main/dense_1/kernel/Adam_1/Initializer/zeros/Const*
T0*&
_class
loc:@main/dense_1/kernel*

index_type0*
_output_shapes
:	@
И
main/dense_1/kernel/Adam_1
VariableV2*
_output_shapes
:	@*
shared_name *&
_class
loc:@main/dense_1/kernel*
	container *
shape:	@*
dtype0
№
!main/dense_1/kernel/Adam_1/AssignAssignmain/dense_1/kernel/Adam_1,main/dense_1/kernel/Adam_1/Initializer/zeros*&
_class
loc:@main/dense_1/kernel*
validate_shape(*
_output_shapes
:	@*
use_locking(*
T0

main/dense_1/kernel/Adam_1/readIdentitymain/dense_1/kernel/Adam_1*
_output_shapes
:	@*
T0*&
_class
loc:@main/dense_1/kernel

(main/dense_1/bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes	
:*$
_class
loc:@main/dense_1/bias*
valueB*    
Њ
main/dense_1/bias/Adam
VariableV2*
_output_shapes	
:*
shared_name *$
_class
loc:@main/dense_1/bias*
	container *
shape:*
dtype0
о
main/dense_1/bias/Adam/AssignAssignmain/dense_1/bias/Adam(main/dense_1/bias/Adam/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@main/dense_1/bias*
validate_shape(*
_output_shapes	
:

main/dense_1/bias/Adam/readIdentitymain/dense_1/bias/Adam*
T0*$
_class
loc:@main/dense_1/bias*
_output_shapes	
:

*main/dense_1/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes	
:*$
_class
loc:@main/dense_1/bias*
valueB*    
Ќ
main/dense_1/bias/Adam_1
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *$
_class
loc:@main/dense_1/bias*
	container *
shape:
ф
main/dense_1/bias/Adam_1/AssignAssignmain/dense_1/bias/Adam_1*main/dense_1/bias/Adam_1/Initializer/zeros*$
_class
loc:@main/dense_1/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0

main/dense_1/bias/Adam_1/readIdentitymain/dense_1/bias/Adam_1*
T0*$
_class
loc:@main/dense_1/bias*
_output_shapes	
:
Г
:main/dense_2/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*&
_class
loc:@main/dense_2/kernel*
valueB"      *
dtype0

0main/dense_2/kernel/Adam/Initializer/zeros/ConstConst*&
_class
loc:@main/dense_2/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

*main/dense_2/kernel/Adam/Initializer/zerosFill:main/dense_2/kernel/Adam/Initializer/zeros/shape_as_tensor0main/dense_2/kernel/Adam/Initializer/zeros/Const*
T0*&
_class
loc:@main/dense_2/kernel*

index_type0* 
_output_shapes
:

И
main/dense_2/kernel/Adam
VariableV2*
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *&
_class
loc:@main/dense_2/kernel*
	container 
ы
main/dense_2/kernel/Adam/AssignAssignmain/dense_2/kernel/Adam*main/dense_2/kernel/Adam/Initializer/zeros*
use_locking(*
T0*&
_class
loc:@main/dense_2/kernel*
validate_shape(* 
_output_shapes
:


main/dense_2/kernel/Adam/readIdentitymain/dense_2/kernel/Adam*
T0*&
_class
loc:@main/dense_2/kernel* 
_output_shapes
:

Е
<main/dense_2/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*&
_class
loc:@main/dense_2/kernel*
valueB"      *
dtype0*
_output_shapes
:

2main/dense_2/kernel/Adam_1/Initializer/zeros/ConstConst*&
_class
loc:@main/dense_2/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

,main/dense_2/kernel/Adam_1/Initializer/zerosFill<main/dense_2/kernel/Adam_1/Initializer/zeros/shape_as_tensor2main/dense_2/kernel/Adam_1/Initializer/zeros/Const*
T0*&
_class
loc:@main/dense_2/kernel*

index_type0* 
_output_shapes
:

К
main/dense_2/kernel/Adam_1
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *&
_class
loc:@main/dense_2/kernel*
	container *
shape:

ё
!main/dense_2/kernel/Adam_1/AssignAssignmain/dense_2/kernel/Adam_1,main/dense_2/kernel/Adam_1/Initializer/zeros*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*&
_class
loc:@main/dense_2/kernel

main/dense_2/kernel/Adam_1/readIdentitymain/dense_2/kernel/Adam_1* 
_output_shapes
:
*
T0*&
_class
loc:@main/dense_2/kernel

(main/dense_2/bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes	
:*$
_class
loc:@main/dense_2/bias*
valueB*    
Њ
main/dense_2/bias/Adam
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *$
_class
loc:@main/dense_2/bias*
	container *
shape:
о
main/dense_2/bias/Adam/AssignAssignmain/dense_2/bias/Adam(main/dense_2/bias/Adam/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*$
_class
loc:@main/dense_2/bias*
validate_shape(

main/dense_2/bias/Adam/readIdentitymain/dense_2/bias/Adam*
_output_shapes	
:*
T0*$
_class
loc:@main/dense_2/bias

*main/dense_2/bias/Adam_1/Initializer/zerosConst*$
_class
loc:@main/dense_2/bias*
valueB*    *
dtype0*
_output_shapes	
:
Ќ
main/dense_2/bias/Adam_1
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *$
_class
loc:@main/dense_2/bias*
	container *
shape:
ф
main/dense_2/bias/Adam_1/AssignAssignmain/dense_2/bias/Adam_1*main/dense_2/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@main/dense_2/bias*
validate_shape(*
_output_shapes	
:

main/dense_2/bias/Adam_1/readIdentitymain/dense_2/bias/Adam_1*$
_class
loc:@main/dense_2/bias*
_output_shapes	
:*
T0
Г
:main/dense_3/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*&
_class
loc:@main/dense_3/kernel*
valueB"      

0main/dense_3/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *&
_class
loc:@main/dense_3/kernel*
valueB
 *    

*main/dense_3/kernel/Adam/Initializer/zerosFill:main/dense_3/kernel/Adam/Initializer/zeros/shape_as_tensor0main/dense_3/kernel/Adam/Initializer/zeros/Const* 
_output_shapes
:
*
T0*&
_class
loc:@main/dense_3/kernel*

index_type0
И
main/dense_3/kernel/Adam
VariableV2*
shared_name *&
_class
loc:@main/dense_3/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:

ы
main/dense_3/kernel/Adam/AssignAssignmain/dense_3/kernel/Adam*main/dense_3/kernel/Adam/Initializer/zeros*&
_class
loc:@main/dense_3/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0

main/dense_3/kernel/Adam/readIdentitymain/dense_3/kernel/Adam* 
_output_shapes
:
*
T0*&
_class
loc:@main/dense_3/kernel
Е
<main/dense_3/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*&
_class
loc:@main/dense_3/kernel*
valueB"      *
dtype0*
_output_shapes
:

2main/dense_3/kernel/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *&
_class
loc:@main/dense_3/kernel*
valueB
 *    

,main/dense_3/kernel/Adam_1/Initializer/zerosFill<main/dense_3/kernel/Adam_1/Initializer/zeros/shape_as_tensor2main/dense_3/kernel/Adam_1/Initializer/zeros/Const* 
_output_shapes
:
*
T0*&
_class
loc:@main/dense_3/kernel*

index_type0
К
main/dense_3/kernel/Adam_1
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *&
_class
loc:@main/dense_3/kernel*
	container *
shape:

ё
!main/dense_3/kernel/Adam_1/AssignAssignmain/dense_3/kernel/Adam_1,main/dense_3/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*&
_class
loc:@main/dense_3/kernel*
validate_shape(* 
_output_shapes
:


main/dense_3/kernel/Adam_1/readIdentitymain/dense_3/kernel/Adam_1*
T0*&
_class
loc:@main/dense_3/kernel* 
_output_shapes
:


(main/dense_3/bias/Adam/Initializer/zerosConst*
_output_shapes	
:*$
_class
loc:@main/dense_3/bias*
valueB*    *
dtype0
Њ
main/dense_3/bias/Adam
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *$
_class
loc:@main/dense_3/bias*
	container *
shape:
о
main/dense_3/bias/Adam/AssignAssignmain/dense_3/bias/Adam(main/dense_3/bias/Adam/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@main/dense_3/bias*
validate_shape(*
_output_shapes	
:

main/dense_3/bias/Adam/readIdentitymain/dense_3/bias/Adam*
_output_shapes	
:*
T0*$
_class
loc:@main/dense_3/bias

*main/dense_3/bias/Adam_1/Initializer/zerosConst*$
_class
loc:@main/dense_3/bias*
valueB*    *
dtype0*
_output_shapes	
:
Ќ
main/dense_3/bias/Adam_1
VariableV2*
shared_name *$
_class
loc:@main/dense_3/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
ф
main/dense_3/bias/Adam_1/AssignAssignmain/dense_3/bias/Adam_1*main/dense_3/bias/Adam_1/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*$
_class
loc:@main/dense_3/bias*
validate_shape(

main/dense_3/bias/Adam_1/readIdentitymain/dense_3/bias/Adam_1*$
_class
loc:@main/dense_3/bias*
_output_shapes	
:*
T0
Љ
*main/dense_4/kernel/Adam/Initializer/zerosConst*&
_class
loc:@main/dense_4/kernel*
valueB	*    *
dtype0*
_output_shapes
:	
Ж
main/dense_4/kernel/Adam
VariableV2*
shared_name *&
_class
loc:@main/dense_4/kernel*
	container *
shape:	*
dtype0*
_output_shapes
:	
ъ
main/dense_4/kernel/Adam/AssignAssignmain/dense_4/kernel/Adam*main/dense_4/kernel/Adam/Initializer/zeros*
_output_shapes
:	*
use_locking(*
T0*&
_class
loc:@main/dense_4/kernel*
validate_shape(

main/dense_4/kernel/Adam/readIdentitymain/dense_4/kernel/Adam*
_output_shapes
:	*
T0*&
_class
loc:@main/dense_4/kernel
Ћ
,main/dense_4/kernel/Adam_1/Initializer/zerosConst*&
_class
loc:@main/dense_4/kernel*
valueB	*    *
dtype0*
_output_shapes
:	
И
main/dense_4/kernel/Adam_1
VariableV2*
shared_name *&
_class
loc:@main/dense_4/kernel*
	container *
shape:	*
dtype0*
_output_shapes
:	
№
!main/dense_4/kernel/Adam_1/AssignAssignmain/dense_4/kernel/Adam_1,main/dense_4/kernel/Adam_1/Initializer/zeros*&
_class
loc:@main/dense_4/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0

main/dense_4/kernel/Adam_1/readIdentitymain/dense_4/kernel/Adam_1*
_output_shapes
:	*
T0*&
_class
loc:@main/dense_4/kernel

(main/dense_4/bias/Adam/Initializer/zerosConst*$
_class
loc:@main/dense_4/bias*
valueB*    *
dtype0*
_output_shapes
:
Ј
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
н
main/dense_4/bias/Adam/AssignAssignmain/dense_4/bias/Adam(main/dense_4/bias/Adam/Initializer/zeros*
T0*$
_class
loc:@main/dense_4/bias*
validate_shape(*
_output_shapes
:*
use_locking(

main/dense_4/bias/Adam/readIdentitymain/dense_4/bias/Adam*
T0*$
_class
loc:@main/dense_4/bias*
_output_shapes
:

*main/dense_4/bias/Adam_1/Initializer/zerosConst*$
_class
loc:@main/dense_4/bias*
valueB*    *
dtype0*
_output_shapes
:
Њ
main/dense_4/bias/Adam_1
VariableV2*
shared_name *$
_class
loc:@main/dense_4/bias*
	container *
shape:*
dtype0*
_output_shapes
:
у
main/dense_4/bias/Adam_1/AssignAssignmain/dense_4/bias/Adam_1*main/dense_4/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@main/dense_4/bias*
validate_shape(*
_output_shapes
:

main/dense_4/bias/Adam_1/readIdentitymain/dense_4/bias/Adam_1*
T0*$
_class
loc:@main/dense_4/bias*
_output_shapes
:
W
Adam/learning_rateConst*
dtype0*
_output_shapes
: *
valueB
 *Зб8
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
 *wО?*
dtype0*
_output_shapes
: 
Q
Adam/epsilonConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 

'Adam/update_main/dense/kernel/ApplyAdam	ApplyAdammain/dense/kernelmain/dense/kernel/Adammain/dense/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon;gradients/main/dense/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*$
_class
loc:@main/dense/kernel*
use_nesterov( *
_output_shapes
:	
ў
%Adam/update_main/dense/bias/ApplyAdam	ApplyAdammain/dense/biasmain/dense/bias/Adammain/dense/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon<gradients/main/dense/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes	
:*
use_locking( *
T0*"
_class
loc:@main/dense/bias*
use_nesterov( 

)Adam/update_main/dense_1/kernel/ApplyAdam	ApplyAdammain/dense_1/kernelmain/dense_1/kernel/Adammain/dense_1/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon=gradients/main/dense_1/MatMul_grad/tuple/control_dependency_1*&
_class
loc:@main/dense_1/kernel*
use_nesterov( *
_output_shapes
:	@*
use_locking( *
T0

'Adam/update_main/dense_1/bias/ApplyAdam	ApplyAdammain/dense_1/biasmain/dense_1/bias/Adammain/dense_1/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon>gradients/main/dense_1/BiasAdd_grad/tuple/control_dependency_1*$
_class
loc:@main/dense_1/bias*
use_nesterov( *
_output_shapes	
:*
use_locking( *
T0

)Adam/update_main/dense_2/kernel/ApplyAdam	ApplyAdammain/dense_2/kernelmain/dense_2/kernel/Adammain/dense_2/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon=gradients/main/dense_2/MatMul_grad/tuple/control_dependency_1* 
_output_shapes
:
*
use_locking( *
T0*&
_class
loc:@main/dense_2/kernel*
use_nesterov( 

'Adam/update_main/dense_2/bias/ApplyAdam	ApplyAdammain/dense_2/biasmain/dense_2/bias/Adammain/dense_2/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon>gradients/main/dense_2/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*$
_class
loc:@main/dense_2/bias*
use_nesterov( *
_output_shapes	
:

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


'Adam/update_main/dense_3/bias/ApplyAdam	ApplyAdammain/dense_3/biasmain/dense_3/bias/Adammain/dense_3/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon>gradients/main/dense_3/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*$
_class
loc:@main/dense_3/bias*
use_nesterov( *
_output_shapes	
:

)Adam/update_main/dense_4/kernel/ApplyAdam	ApplyAdammain/dense_4/kernelmain/dense_4/kernel/Adammain/dense_4/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon=gradients/main/dense_4/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*&
_class
loc:@main/dense_4/kernel*
use_nesterov( *
_output_shapes
:	

'Adam/update_main/dense_4/bias/ApplyAdam	ApplyAdammain/dense_4/biasmain/dense_4/bias/Adammain/dense_4/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon>gradients/main/dense_4/BiasAdd_grad/tuple/control_dependency_1*
T0*$
_class
loc:@main/dense_4/bias*
use_nesterov( *
_output_shapes
:*
use_locking( 

Adam/mulMulbeta1_power/read
Adam/beta1&^Adam/update_main/dense/bias/ApplyAdam(^Adam/update_main/dense/kernel/ApplyAdam(^Adam/update_main/dense_1/bias/ApplyAdam*^Adam/update_main/dense_1/kernel/ApplyAdam(^Adam/update_main/dense_2/bias/ApplyAdam*^Adam/update_main/dense_2/kernel/ApplyAdam(^Adam/update_main/dense_3/bias/ApplyAdam*^Adam/update_main/dense_3/kernel/ApplyAdam(^Adam/update_main/dense_4/bias/ApplyAdam*^Adam/update_main/dense_4/kernel/ApplyAdam*
T0*"
_class
loc:@main/dense/bias*
_output_shapes
: 

Adam/AssignAssignbeta1_powerAdam/mul*
T0*"
_class
loc:@main/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking( 


Adam/mul_1Mulbeta2_power/read
Adam/beta2&^Adam/update_main/dense/bias/ApplyAdam(^Adam/update_main/dense/kernel/ApplyAdam(^Adam/update_main/dense_1/bias/ApplyAdam*^Adam/update_main/dense_1/kernel/ApplyAdam(^Adam/update_main/dense_2/bias/ApplyAdam*^Adam/update_main/dense_2/kernel/ApplyAdam(^Adam/update_main/dense_3/bias/ApplyAdam*^Adam/update_main/dense_3/kernel/ApplyAdam(^Adam/update_main/dense_4/bias/ApplyAdam*^Adam/update_main/dense_4/kernel/ApplyAdam*
_output_shapes
: *
T0*"
_class
loc:@main/dense/bias

Adam/Assign_1Assignbeta2_power
Adam/mul_1*
use_locking( *
T0*"
_class
loc:@main/dense/bias*
validate_shape(*
_output_shapes
: 
д
AdamNoOp^Adam/Assign^Adam/Assign_1&^Adam/update_main/dense/bias/ApplyAdam(^Adam/update_main/dense/kernel/ApplyAdam(^Adam/update_main/dense_1/bias/ApplyAdam*^Adam/update_main/dense_1/kernel/ApplyAdam(^Adam/update_main/dense_2/bias/ApplyAdam*^Adam/update_main/dense_2/kernel/ApplyAdam(^Adam/update_main/dense_3/bias/ApplyAdam*^Adam/update_main/dense_3/kernel/ApplyAdam(^Adam/update_main/dense_4/bias/ApplyAdam*^Adam/update_main/dense_4/kernel/ApplyAdam
И
AssignAssigntarget/dense/kernelmain/dense/kernel/read*
use_locking(*
T0*&
_class
loc:@target/dense/kernel*
validate_shape(*
_output_shapes
:	
А
Assign_1Assigntarget/dense/biasmain/dense/bias/read*
use_locking(*
T0*$
_class
loc:@target/dense/bias*
validate_shape(*
_output_shapes	
:
Р
Assign_2Assigntarget/dense_1/kernelmain/dense_1/kernel/read*
use_locking(*
T0*(
_class
loc:@target/dense_1/kernel*
validate_shape(*
_output_shapes
:	@
Ж
Assign_3Assigntarget/dense_1/biasmain/dense_1/bias/read*
use_locking(*
T0*&
_class
loc:@target/dense_1/bias*
validate_shape(*
_output_shapes	
:
С
Assign_4Assigntarget/dense_2/kernelmain/dense_2/kernel/read*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*(
_class
loc:@target/dense_2/kernel
Ж
Assign_5Assigntarget/dense_2/biasmain/dense_2/bias/read*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*&
_class
loc:@target/dense_2/bias
С
Assign_6Assigntarget/dense_3/kernelmain/dense_3/kernel/read*
T0*(
_class
loc:@target/dense_3/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
Ж
Assign_7Assigntarget/dense_3/biasmain/dense_3/bias/read*
use_locking(*
T0*&
_class
loc:@target/dense_3/bias*
validate_shape(*
_output_shapes	
:
Р
Assign_8Assigntarget/dense_4/kernelmain/dense_4/kernel/read*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*(
_class
loc:@target/dense_4/kernel
Е
Assign_9Assigntarget/dense_4/biasmain/dense_4/bias/read*
use_locking(*
T0*&
_class
loc:@target/dense_4/bias*
validate_shape(*
_output_shapes
:


initNoOp^beta1_power/Assign^beta2_power/Assign^main/dense/bias/Adam/Assign^main/dense/bias/Adam_1/Assign^main/dense/bias/Assign^main/dense/kernel/Adam/Assign ^main/dense/kernel/Adam_1/Assign^main/dense/kernel/Assign^main/dense_1/bias/Adam/Assign ^main/dense_1/bias/Adam_1/Assign^main/dense_1/bias/Assign ^main/dense_1/kernel/Adam/Assign"^main/dense_1/kernel/Adam_1/Assign^main/dense_1/kernel/Assign^main/dense_2/bias/Adam/Assign ^main/dense_2/bias/Adam_1/Assign^main/dense_2/bias/Assign ^main/dense_2/kernel/Adam/Assign"^main/dense_2/kernel/Adam_1/Assign^main/dense_2/kernel/Assign^main/dense_3/bias/Adam/Assign ^main/dense_3/bias/Adam_1/Assign^main/dense_3/bias/Assign ^main/dense_3/kernel/Adam/Assign"^main/dense_3/kernel/Adam_1/Assign^main/dense_3/kernel/Assign^main/dense_4/bias/Adam/Assign ^main/dense_4/bias/Adam_1/Assign^main/dense_4/bias/Assign ^main/dense_4/kernel/Adam/Assign"^main/dense_4/kernel/Adam_1/Assign^main/dense_4/kernel/Assign^target/dense/bias/Assign^target/dense/kernel/Assign^target/dense_1/bias/Assign^target/dense_1/kernel/Assign^target/dense_2/bias/Assign^target/dense_2/kernel/Assign^target/dense_3/bias/Assign^target/dense_3/kernel/Assign^target/dense_4/bias/Assign^target/dense_4/kernel/Assign
R
Placeholder_4Placeholder*
shape:*
dtype0*
_output_shapes
:
R
reward/tagsConst*
valueB Breward*
dtype0*
_output_shapes
: 
T
rewardScalarSummaryreward/tagsPlaceholder_4*
T0*
_output_shapes
: 
K
Merge/MergeSummaryMergeSummaryreward*
N*
_output_shapes
: "" 
losses

huber_loss/Mul_3:0"
	summaries


reward:0"
trainable_variablesіѓ
{
main/dense/kernel:0main/dense/kernel/Assignmain/dense/kernel/read:02.main/dense/kernel/Initializer/random_uniform:08
j
main/dense/bias:0main/dense/bias/Assignmain/dense/bias/read:02#main/dense/bias/Initializer/zeros:08

main/dense_1/kernel:0main/dense_1/kernel/Assignmain/dense_1/kernel/read:020main/dense_1/kernel/Initializer/random_uniform:08
r
main/dense_1/bias:0main/dense_1/bias/Assignmain/dense_1/bias/read:02%main/dense_1/bias/Initializer/zeros:08

main/dense_2/kernel:0main/dense_2/kernel/Assignmain/dense_2/kernel/read:020main/dense_2/kernel/Initializer/random_uniform:08
r
main/dense_2/bias:0main/dense_2/bias/Assignmain/dense_2/bias/read:02%main/dense_2/bias/Initializer/zeros:08

main/dense_3/kernel:0main/dense_3/kernel/Assignmain/dense_3/kernel/read:020main/dense_3/kernel/Initializer/random_uniform:08
r
main/dense_3/bias:0main/dense_3/bias/Assignmain/dense_3/bias/read:02%main/dense_3/bias/Initializer/zeros:08

main/dense_4/kernel:0main/dense_4/kernel/Assignmain/dense_4/kernel/read:020main/dense_4/kernel/Initializer/random_uniform:08
r
main/dense_4/bias:0main/dense_4/bias/Assignmain/dense_4/bias/read:02%main/dense_4/bias/Initializer/zeros:08

target/dense/kernel:0target/dense/kernel/Assigntarget/dense/kernel/read:020target/dense/kernel/Initializer/random_uniform:08
r
target/dense/bias:0target/dense/bias/Assigntarget/dense/bias/read:02%target/dense/bias/Initializer/zeros:08

target/dense_1/kernel:0target/dense_1/kernel/Assigntarget/dense_1/kernel/read:022target/dense_1/kernel/Initializer/random_uniform:08
z
target/dense_1/bias:0target/dense_1/bias/Assigntarget/dense_1/bias/read:02'target/dense_1/bias/Initializer/zeros:08

target/dense_2/kernel:0target/dense_2/kernel/Assigntarget/dense_2/kernel/read:022target/dense_2/kernel/Initializer/random_uniform:08
z
target/dense_2/bias:0target/dense_2/bias/Assigntarget/dense_2/bias/read:02'target/dense_2/bias/Initializer/zeros:08

target/dense_3/kernel:0target/dense_3/kernel/Assigntarget/dense_3/kernel/read:022target/dense_3/kernel/Initializer/random_uniform:08
z
target/dense_3/bias:0target/dense_3/bias/Assigntarget/dense_3/bias/read:02'target/dense_3/bias/Initializer/zeros:08

target/dense_4/kernel:0target/dense_4/kernel/Assigntarget/dense_4/kernel/read:022target/dense_4/kernel/Initializer/random_uniform:08
z
target/dense_4/bias:0target/dense_4/bias/Assigntarget/dense_4/bias/read:02'target/dense_4/bias/Initializer/zeros:08"
train_op

Adam"Л+
	variables­+Њ+
{
main/dense/kernel:0main/dense/kernel/Assignmain/dense/kernel/read:02.main/dense/kernel/Initializer/random_uniform:08
j
main/dense/bias:0main/dense/bias/Assignmain/dense/bias/read:02#main/dense/bias/Initializer/zeros:08

main/dense_1/kernel:0main/dense_1/kernel/Assignmain/dense_1/kernel/read:020main/dense_1/kernel/Initializer/random_uniform:08
r
main/dense_1/bias:0main/dense_1/bias/Assignmain/dense_1/bias/read:02%main/dense_1/bias/Initializer/zeros:08

main/dense_2/kernel:0main/dense_2/kernel/Assignmain/dense_2/kernel/read:020main/dense_2/kernel/Initializer/random_uniform:08
r
main/dense_2/bias:0main/dense_2/bias/Assignmain/dense_2/bias/read:02%main/dense_2/bias/Initializer/zeros:08

main/dense_3/kernel:0main/dense_3/kernel/Assignmain/dense_3/kernel/read:020main/dense_3/kernel/Initializer/random_uniform:08
r
main/dense_3/bias:0main/dense_3/bias/Assignmain/dense_3/bias/read:02%main/dense_3/bias/Initializer/zeros:08

main/dense_4/kernel:0main/dense_4/kernel/Assignmain/dense_4/kernel/read:020main/dense_4/kernel/Initializer/random_uniform:08
r
main/dense_4/bias:0main/dense_4/bias/Assignmain/dense_4/bias/read:02%main/dense_4/bi
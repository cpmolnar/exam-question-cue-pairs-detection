# Exam Question Cue Detection With BERT-SQuAD
A project leveraging Huggingface Transformers' pre-trained BERT-SQuAD Question Answering model to detect "cue" questions within an exam question database.  
Cue questions are pairs of questions whereby knowing the answer to one question reveals the answer to another.  
The approach I took is:  
- Search for pairs of questions with the same answer.
- Rearrange one potential cue, and four other random questions, into statements.
- Randomly arrange the five questions into a block of text.
- Use the trained model to search for the answer to the given question within the generated text.

This roughly simulates the trained environment of the BERT-SQuAD model.  
Model accuracy is inconclusive, but it certainly achieves interesting results fairly quickly on my very modest hardware.  
Could warrant further investigation with a knowledgeable team to analyze results, or labelled data.  

Of course, cues can also be detected by revealing contextual information of another question, but this is far outside the scope of what this model can achieve.  

# Requirements
- python3
- pip3 install -r requirements.txt

# Sample Results (High Confidence)
progress: 621 / 81810  
text: the posterior circulation of the brain is best demonstrated by injecting contrast material into the vertebral artery.  
blood from the inferior vena cava enters the heart through the right atrium.  
there are no lymphatic capillaries in the brain.  
amaurosis fugax may result from a stenosis of common carotid artery.  
a cholesteatoma is a pathologic process requiring a ct scan of the middle ear.  
question: there are no lymphatic capillaries in the?  
question answer: brain  
prediction: brain  
start: 37  
end: 37  
confidence: 0.8210961429022994  

progress: 3633 / 81810  
text:   
pituitary gland is situated in the sella turcica.   
anterior communicating is an intracranial artery.   
8 cranial bones make up the human skull.   
on a lateral projection of a cerebral angiogram the large vessel draining from anterior to posterior along the curve of the cranial bone is the superior sagittal sinus.   
sigmoid sinus drains venous blood directly from the brain to the internal jugular vein.  
question: what sphenoid bone structure contains the pituitary gland?  
question answer: sella turcica  
prediction: sella turcica  
start: 6  
end: 7  
confidence: 0.8438154495784939  

# Author  
Carl Molnar  
3/12/2020
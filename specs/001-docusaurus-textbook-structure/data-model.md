# Data Model: Physical AI & Humanoid Robotics Textbook

## Entities

### Chapter
- **id**: string (unique identifier)
- **title**: string (chapter title)
- **module**: string (module identifier, e.g. "module-1")
- **week**: number (week number within module)
- **content**: string (MDX content)
- **objectives**: array of string (learning objectives)
- **codeExamples**: array of string (code examples in the chapter)
- **exercises**: array of object (exercise definitions)
- **visuals**: array of string (image/video references)
- **createdAt**: datetime
- **updatedAt**: datetime
- **status**: enum (draft, review, published)

### Module
- **id**: string (unique identifier, e.g. "module-1")
- **title**: string (module title)
- **description**: string (module overview)
- **weeks**: number (number of weeks in module)
- **chapters**: array of Chapter (related chapters)
- **createdAt**: datetime
- **updatedAt**: datetime
- **status**: enum (draft, review, published)

### User
- **id**: string (unique identifier)
- **email**: string (user's email, unique)
- **name**: string (user's full name)
- **role**: enum (student, instructor, admin)
- **preferences**: object (user preferences for personalization)
- **progress**: object (tracking progress through chapters/modules)
- **createdAt**: datetime
- **updatedAt**: datetime
- **lastLoginAt**: datetime

### Progress
- **id**: string (unique identifier)
- **userId**: string (foreign key to User)
- **chapterId**: string (foreign key to Chapter)
- **completed**: boolean (whether chapter is completed)
- **completionDate**: datetime (when chapter was completed)
- **timeSpent**: number (time spent on chapter in seconds)
- **quizResults**: array of object (results of any quizzes)
- **personalizationData**: object (data for personalization engine)
- **createdAt**: datetime
- **updatedAt**: datetime

### ChatSession
- **id**: string (unique identifier)
- **userId**: string (foreign key to User, nullable for anonymous)
- **messages**: array of object (chat message history)
- **context**: object (current context for the conversation)
- **createdAt**: datetime
- **updatedAt**: datetime
- **isActive**: boolean

### Translation
- **id**: string (unique identifier)
- **sourceId**: string (ID of the source content)
- **sourceType**: enum (chapter, module, page)
- **language**: string (language code, e.g. "ur" for Urdu)
- **content**: string (translated content)
- **status**: enum (pending, in-progress, completed, reviewed)
- **translator**: string (who translated)
- **createdAt**: datetime
- **updatedAt**: datetime

## Relationships

- Module **has many** Chapters
- User **has many** Progress records
- User **has many** ChatSessions (optional, for logged-in users)
- Chapter **has many** Progress records (one per user who accesses it)
- ChatSession **belongs to** User (optional for anonymous sessions)

## Validation Rules

- User email must be unique and valid format
- Chapter title and content are required
- Module weeks must be between 1 and 10
- Progress completion status can only be updated by the system or user
- Translation language must be supported by the system

## State Transitions

- Chapter: draft → review → published
- Translation: pending → in-progress → completed → reviewed
- User role can be updated by admin users only
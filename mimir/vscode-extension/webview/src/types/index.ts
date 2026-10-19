// The two unions every message belongs to, and every type of the panel, from one import.

export * from "./connection";
export * from "./settings";
export * from "./turn";
export * from "./tools";
export * from "./prompts";
export * from "./jobs";
export * from "./session";
export * from "./panels";
export * from "./editor";
export * from "./chat";

import type { AutoConnectMessage, ConfigMessage, DetachMessage, DetachedMessage, ModelChangedMessage, ModelsMessage, ReadyMessage, ServedModelsMessage, SetModelMessage, ShutdownMessage } from "./connection";
import type { ActiveEditorMessage, FileResultsMessage, ListResourcesMessage, OpenEditorMessage, ResourcesMessage } from "./editor";
import type { JobCheckinMessage, JobCompleteMessage, JobProgressMessage } from "./jobs";
import type { ListPanelMessage, ListSubagentsMessage, PanelReportMessage, ReadSubagentMessage, SubAgentMessage, SubAgentsListMessage } from "./panels";
import type { ApprovalMessage, ApprovalResponseMessage, BatchReviewAcceptFileMessage, BatchReviewAcceptMessage, BatchReviewRevertFileMessage, BatchReviewRevertMessage, BatchStatusMessage, PromptExpiredMessage, UserQuestionMessage, UserQuestionResponseMessage } from "./prompts";
import type { ClearTodosMessage, CreateSessionMessage, DeleteSessionMessage, RenameSessionMessage, ReplayMessage, SessionLoadedMessage, SessionsListMessage, SwitchSessionMessage, TodoMessage, TranscriptMessage, TranscriptResyncMessage } from "./session";
import type { AgentModeMessage, ApprovalModeMessage, ContextModeMessage, EnforcementModeMessage, ListTogglesMessage, StreamingStateMessage, SubAgentLevelMessage, TemperatureMessage, ThinkingDepthMessage, ToggleServerMessage, ToggleSkillMessage, TogglesListMessage } from "./settings";
import type { DiffMessage, DivertToBackgroundMessage, FileProgressMessage, SubAgentEventMessage, ToolBackgroundedMessage, ToolCallMessage, ToolProgressMessage, ToolResultMessage, VerdictMessage } from "./tools";
import type { AnswerMessage, AssistantTextMessage, CommandMessage, CommandOutputMessage, ContextUsageMessage, ErrorMessage, NudgeInjectedMessage, OutputMessage, QueryMessage, QueuedMessage, StatusMessage, SteerInjectedMessage, SteerMessage, ThinkingEndMessage, ThinkingMessage, ThinkingStartMessage, TokenMessage } from "./turn";

export type ServerMessage =
  | ReadyMessage
  | OutputMessage
  | StatusMessage
  | CommandOutputMessage
  | ApprovalMessage
  | AnswerMessage
  | TokenMessage
  | TodoMessage
  | ThinkingMessage
  | ThinkingStartMessage
  | ThinkingEndMessage
  | ToolCallMessage
  | ToolResultMessage
  | ToolProgressMessage
  | ToolBackgroundedMessage
  | JobProgressMessage
  | VerdictMessage
  | SubAgentEventMessage
  | ErrorMessage
  | ConfigMessage
  | AutoConnectMessage
  | ModelsMessage
  | ServedModelsMessage
  | ModelChangedMessage
  | DiffMessage
  | SteerInjectedMessage
  | NudgeInjectedMessage
  | FileProgressMessage
  | BatchStatusMessage
  | SessionsListMessage
  | SubAgentsListMessage
  | SubAgentMessage
  | SubAgentLevelMessage
  | PanelReportMessage
  | SessionLoadedMessage
  | ContextModeMessage
  | EnforcementModeMessage
  | ApprovalModeMessage
  | AgentModeMessage
  | ThinkingDepthMessage
  | TemperatureMessage
  | StreamingStateMessage
  | ContextUsageMessage
  | JobCompleteMessage
  | JobCheckinMessage
  | TogglesListMessage
  | ResourcesMessage
  | FileResultsMessage
  | ActiveEditorMessage
  | OpenEditorMessage
  | UserQuestionMessage
  | PromptExpiredMessage
  | ReplayMessage
  | AssistantTextMessage
  | DetachedMessage
  | TranscriptResyncMessage
  | QueuedMessage;

export type ClientMessage =
  | QueryMessage
  | ListSubagentsMessage
  | ListPanelMessage
  | ReadSubagentMessage
  | ClearTodosMessage
  | TranscriptMessage
  | SteerMessage
  | DetachMessage
  | DivertToBackgroundMessage
  | ApprovalResponseMessage
  | UserQuestionResponseMessage
  | CommandMessage
  | CreateSessionMessage
  | SwitchSessionMessage
  | DeleteSessionMessage
  | RenameSessionMessage
  | BatchReviewAcceptMessage
  | BatchReviewRevertMessage
  | BatchReviewAcceptFileMessage
  | BatchReviewRevertFileMessage
  | ListTogglesMessage
  | ListResourcesMessage
  | ToggleServerMessage
  | ToggleSkillMessage
  | SetModelMessage
  | ShutdownMessage;

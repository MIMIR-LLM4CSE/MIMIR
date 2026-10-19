// The connect form, seeded from what the endpoint hook knows: the settings' defaults,
// the remembered address and the models the endpoint reported.

import React from "react";
import type { useEndpoint } from "../hooks/useEndpoint";
import { ConnectForm } from "./ConnectForm";

export const EndpointForm: React.FC<{ endpoint: ReturnType<typeof useEndpoint> }> = ({ endpoint }) => (
  <ConnectForm
    key={endpoint.connectFormKey}
    backend={endpoint.backend}
    vllmBaseUrl={endpoint.vllmBaseUrl}
    rayBaseUrl={endpoint.rayBaseUrl}
    ollamaBaseUrl={endpoint.ollamaBaseUrl}
    anthropicModels={endpoint.anthropicModels}
    models={endpoint.endpointModels}
    modelsError={endpoint.modelsError}
    modelsProbed={endpoint.modelsProbed}
    remembered={endpoint.remembered}
    onFetchModels={endpoint.fetchModels}
    onConnect={endpoint.connect}
  />
);

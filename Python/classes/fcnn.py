import torch.nn as nn


class Fcnn(nn.Module):
    def __init__(self, input_layer_size=64*6, output_layer_size=64, hidden_layers_size=750):
        super(Fcnn, self).__init__()

        self.input_to_hidden_layer = nn.Linear(input_layer_size, hidden_layers_size)
        self.hidden_to_hidden_layer = nn.Linear(hidden_layers_size, hidden_layers_size)
        self.hidden_to_output_layer = nn.Linear(hidden_layers_size, output_layer_size)
        self.batch_norm_hidden = nn.BatchNorm1d(hidden_layers_size)
        self.batch_norm_output = nn.BatchNorm1d(output_layer_size)
        self.act_layer = nn.ReLU()
        self.dropout_layer = nn.Dropout(0.5)

        self.output_layer = nn.Linear(hidden_layers_size, output_layer_size)

    def forward(self, x):

        input_layer = self.input_to_hidden_layer(x)
        batch_norm_input = self.batch_norm_hidden(input_layer)
        act_input = self.act_layer(batch_norm_input)
        dropout_input = self.dropout_layer(act_input)

        hidden_layer = self.hidden_to_hidden_layer(dropout_input)
        batch_norm_hidden = self.batch_norm_hidden(hidden_layer)
        act_hidden = self.act_layer(batch_norm_hidden)
        dropout_hidden = self.dropout_layer(act_hidden)

        hidden_layer = self.hidden_to_hidden_layer(dropout_hidden)
        batch_norm_hidden = self.batch_norm_hidden(hidden_layer)
        act_hidden = self.act_layer(batch_norm_hidden)
        dropout_hidden = self.dropout_layer(act_hidden)

        output_layer = self.hidden_to_output_layer(dropout_hidden)
        batch_norm_output = self.batch_norm_output(output_layer)
        act_output = self.act_layer(batch_norm_output)

        return act_output


if __name__ == '__main__':
    print('check up')